# app/services/chat_service.py
import uuid
import base64
import tempfile
import os
import re
import mimetypes
import traceback
from typing import List, Optional, Dict, Any

import aiosqlite # Needed for type hinting db parameter
from fastapi import HTTPException

# Assuming prompts.py is accessible in the top-level directory or moved to app/
try:
    from ..prompts import prompts
    PROMPTS_LOADED = True
except ImportError:
    print("WARNING: services.chat_service - prompts.py not found or import failed. Using placeholder prompts.")
    class MockPrompts: # Minimal class definition
        code = "Placeholder Code Prompt - prompts.py not loaded"
        architect = "Placeholder Architect Prompt - prompts.py not loaded"
        debug = "Placeholder Debug Prompt - prompts.py not loaded"
        ask = "Placeholder Ask Prompt - prompts.py not loaded"
    prompts = MockPrompts()
    PROMPTS_LOADED = False


from app.repositories.chat_repository import SqliteChatRepository
from app.core.gemini_client import GeminiClientWrapper
from app.models import ChatInfo, OpenAIMessage, TextBlock, ImageUrlBlock, ChatCompletionResponse, Choice, Usage
from app.config import ALLOWED_MODES, GEMINI_MODEL_NAME

# Mapping from mode names to the actual prompt variables/text
#
# Built-in Roo Code modes have hardcoded prompts for web UI fallback.
# Marketplace modes and "Default" use None because:
#   - When used with Roo Code/Kilo Code: prompts are passed through automatically
#   - When used with web UI: these modes act as simple pass-through (no system prompt)
#
MODE_PROMPT_TEXTS: Dict[str, Optional[str]] = {
    # Built-in modes with hardcoded prompts (for web UI fallback)
    "Code": getattr(prompts, 'code', None),
    "Architect": getattr(prompts, 'architect', None),
    "Debug": getattr(prompts, 'debug', None),
    "Ask": getattr(prompts, 'ask', None),
    
    # Pass-through modes (no hardcoded prompt - Roo Code/Kilo Code sends their own)
    "Default": None,
    "Orchestrator": None,  # Built-in Roo Code mode
    
    # Marketplace modes (all pass-through - prompts come from Roo Code)
    "ModeWriter": None,
    "DocumentationWriter": None,
    "UserStoryCreator": None,
    "ProjectResearch": None,
    "SecurityReview": None,
    "DevOps": None,
    "JestTestEngineer": None,
    "GoogleGenAIDeveloper": None,
    "CodingTeacher": None,
    "GitMergeResolver": None,
}
# Check if any actual prompts failed to load *if* the import was expected to succeed
if PROMPTS_LOADED and None in [MODE_PROMPT_TEXTS.get(m) for m in ["Code", "Architect", "Debug", "Ask"]]:
     print("WARNING: services.chat_service - prompts.py loaded, but one or more specific prompt variables (code, architect, etc.) are missing!")


class ChatService:
    """Orchestrates chat operations, managing state, repository, and Gemini client interactions."""

    def __init__(self, repository: SqliteChatRepository, gemini_wrapper: GeminiClientWrapper):
        self.repository = repository
        self.gemini_wrapper = gemini_wrapper
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._active_chat_id: Optional[str] = None
        print("ChatService initialized.")

    async def load_initial_cache(self, db: aiosqlite.Connection):
        """Loads all session data from DB into the cache."""
        print("ChatService: Loading initial cache from database...")
        try:
            self._cache = await self.repository.get_all_session_data(db)
            print(f"ChatService: Initial cache loaded with {len(self._cache)} sessions.")
        except Exception as e:
             print(f"ChatService CRITICAL ERROR: Failed to load initial cache: {e}")
             self._cache = {}

    async def list_chats(self, db: aiosqlite.Connection) -> List[ChatInfo]:
        """Lists all available chat sessions."""
        return await self.repository.get_chat_info_list(db)

    async def create_chat(self, db: aiosqlite.Connection, description: Optional[str], mode: Optional[ALLOWED_MODES]) -> str:
        """Creates a new chat session, saves it, and updates the cache."""
        new_chat_id = str(uuid.uuid4())
        final_mode = mode or "Default"
        if final_mode not in MODE_PROMPT_TEXTS:
             print(f"Service Warning: Invalid mode '{final_mode}' provided during chat creation. Forcing 'Default'.")
             final_mode = "Default"
        print(f"Service: Creating chat - ID: {new_chat_id}, Desc: '{description or 'N/A'}', Mode: '{final_mode}'")
        try:
            chat_session = self.gemini_wrapper.start_new_chat()
            initial_metadata = chat_session.metadata
            success_db = await self.repository.create_chat(db, new_chat_id, initial_metadata, description, final_mode)
            if not success_db:
                 raise HTTPException(status_code=500, detail="Failed to save new chat session to database (likely already exists or DB error).")
            self._cache[new_chat_id] = {
                "metadata": initial_metadata,
                "mode": final_mode,
                "prompt_sent": False # System prompt NOT sent on creation
            }
            print(f"Service: Chat {new_chat_id} created and added to cache.")
            return new_chat_id
        except Exception as e:
            print(f"Service Error creating chat: {e}")
            traceback.print_exc()
            if isinstance(e, HTTPException): raise e
            raise HTTPException(status_code=500, detail=f"Unexpected error creating chat session: {e}")

    async def set_active_chat(self, db: aiosqlite.Connection, chat_id: Optional[str]):
        """
        Sets the globally active chat ID. If activating a chat, sends the
        system prompt via Gemini if it hasn't been sent for the current mode yet,
        then updates state in DB and cache.
        """
        # --- Deactivation ---
        if chat_id is None:
            if self._active_chat_id is not None:
                print(f"Service: Deactivating active chat {self._active_chat_id}.")
            self._active_chat_id = None
            return

        # --- Activation ---
        print(f"Service: Attempting to activate chat: {chat_id}")

        # 1. Validate chat exists in cache
        if chat_id not in self._cache:
            print(f"Service ERROR: Cannot activate chat - ID '{chat_id}' not found in cache.")
            raise HTTPException(status_code=404, detail=f"Chat session not found in active cache: {chat_id}")

        session_data = self._cache[chat_id]
        metadata = session_data.get("metadata")
        mode = session_data.get("mode", "Default")
        prompt_sent = session_data.get("prompt_sent", False) # Default to False if missing
        system_prompt = MODE_PROMPT_TEXTS.get(mode)

        # Check if metadata exists (essential for sending prompt)
        if metadata is None:
            print(f"Service CRITICAL ERROR: Metadata missing in cache for chat {chat_id} during activation!")
            raise HTTPException(status_code=500, detail="Internal Error: Cannot activate chat, state corrupted.")

        # 2. Send System Prompt if Needed
        prompt_send_error = False
        prompt_sent_this_activation = False # Track if we attempted send in this call
        if system_prompt and not prompt_sent:
            print(f"Service: Activating chat {chat_id}: System prompt needed (Mode: {mode}). Sending...")
            prompt_sent_this_activation = True
            try:
                # Load session, send prompt, get updated metadata
                chat_session = self.gemini_wrapper.load_chat_from_metadata(metadata=metadata)
                await self.gemini_wrapper.send_message(chat_session, system_prompt)
                updated_metadata = chat_session.metadata
                print(f"Service: System prompt sent successfully for {chat_id}.")

                # Update DB: metadata and mark prompt as sent
                print(f"Service: Updating DB for chat {chat_id} post-prompt send...")
                meta_ok = await self.repository.update_metadata(db, chat_id, updated_metadata)
                flag_ok = await self.repository.mark_prompt_sent(db, chat_id)

                # Update cache based on DB success
                if meta_ok:
                    self._cache[chat_id]["metadata"] = updated_metadata
                    print("Service: Metadata cache updated.")
                else:
                    print(f"Service ERROR: Failed to update metadata in DB for {chat_id} post-prompt send. Cache metadata may be stale.")
                    prompt_send_error = True

                if flag_ok:
                    self._cache[chat_id]["prompt_sent"] = True
                    print("Service: prompt_sent flag cache updated.")
                else:
                    print(f"Service ERROR: Failed to mark prompt sent flag in DB for {chat_id}. Cache flag not updated.")
                    prompt_send_error = True

            except Exception as send_error:
                print(f"Service ERROR sending system prompt during activation for {chat_id}: {send_error}")
                traceback.print_exc()
                prompt_send_error = True

        # 3. Set Active ID in memory
        self._active_chat_id = chat_id
        print(f"Service: Active chat set to {self._active_chat_id}")

        if prompt_sent_this_activation and prompt_send_error:
            print(f"Service WARNING: Chat {chat_id} activated, but there was an error sending/confirming the system prompt send state.")
        elif prompt_sent_this_activation:
             print(f"Service: System prompt sending process completed for chat {chat_id} activation.")

    def get_active_chat(self) -> Optional[str]:
        """Gets the currently active chat ID."""
        return self._active_chat_id

    # --- MODIFIED update_chat_mode ---
    async def update_chat_mode(self, db: aiosqlite.Connection, chat_id: str, new_mode: ALLOWED_MODES):
        """
        Updates the mode for a chat, resets the prompt flag in DB/cache, AND
        sends the new system prompt immediately if the updated chat is the active one.
        """
        print(f"Service: Updating mode for chat {chat_id} to '{new_mode}'")
        # Validate mode
        if new_mode not in MODE_PROMPT_TEXTS:
             print(f"Service Warning: Invalid mode '{new_mode}' passed to update_chat_mode.")
             raise HTTPException(status_code=422, detail=f"Invalid mode provided: {new_mode}")
        # Validate chat exists
        if chat_id not in self._cache:
             print(f"Service ERROR: Chat {chat_id} not found in cache for mode update.")
             raise HTTPException(status_code=404, detail="Chat session not found.")

        # 1. Update DB (mode and resets prompt_sent flag) via repository
        success_db = await self.repository.update_mode_and_reset_flag(db, chat_id, new_mode)
        if not success_db:
            print(f"Service Error: DB update failed for mode change on chat {chat_id}.")
            raise HTTPException(status_code=500, detail=f"Failed to update chat mode in database for {chat_id}.")

        # 2. Update cache (mode and prompt_sent flag)
        self._cache[chat_id]["mode"] = new_mode
        self._cache[chat_id]["prompt_sent"] = False # Reset flag first in cache
        print(f"Service: Cache updated for chat {chat_id}: Mode='{new_mode}', prompt_sent=False (reset)")

        # 3. Send new system prompt IF this is the active chat
        if chat_id == self._active_chat_id:
            print(f"Service: Chat {chat_id} is active. Sending system prompt for new mode '{new_mode}'...")
            new_system_prompt = MODE_PROMPT_TEXTS.get(new_mode)

            if new_system_prompt:
                prompt_send_error = False
                try:
                    metadata = self._cache[chat_id].get("metadata")
                    if not metadata:
                         # This should ideally not happen if cache is consistent
                         print(f"Service CRITICAL ERROR: Metadata missing for active chat {chat_id} during mode change prompt send!")
                         raise Exception("Cannot send prompt, metadata missing in cache.")

                    # Load session, send prompt, get updated metadata
                    chat_session = self.gemini_wrapper.load_chat_from_metadata(metadata=metadata)
                    await self.gemini_wrapper.send_message(chat_session, new_system_prompt)
                    updated_metadata = chat_session.metadata
                    print(f"Service: System prompt for new mode '{new_mode}' sent successfully for active chat {chat_id}.")

                    # Update DB: metadata and mark prompt as sent (for the NEW mode)
                    print(f"Service: Updating DB state for chat {chat_id} after immediate prompt send...")
                    meta_ok = await self.repository.update_metadata(db, chat_id, updated_metadata)
                    # Use mark_prompt_sent to set the flag to TRUE now
                    flag_ok = await self.repository.mark_prompt_sent(db, chat_id)

                    # Update cache based on DB success
                    if meta_ok:
                         self._cache[chat_id]["metadata"] = updated_metadata
                         print("Service: Metadata cache updated.")
                    else:
                         prompt_send_error = True
                         print(f"Service ERROR: Failed to update metadata post-mode-change prompt send for {chat_id}.")

                    if flag_ok:
                         self._cache[chat_id]["prompt_sent"] = True # Mark as sent in cache
                         print("Service: prompt_sent flag cache updated to True.")
                    else:
                         prompt_send_error = True
                         # If flag update failed, cache remains False (conservative)
                         print(f"Service ERROR: Failed to mark prompt sent flag post-mode-change prompt send for {chat_id}. Cache flag remains False.")

                    if not prompt_send_error:
                        print(f"Service: DB and cache updated successfully after mode change prompt send for {chat_id}.")

                except Exception as send_error:
                    print(f"Service ERROR sending system prompt during mode change for active chat {chat_id}: {send_error}")
                    traceback.print_exc()
                    prompt_send_error = True
                    # Log the error, but don't raise HTTPException to keep API responsive
                finally:
                     if prompt_send_error:
                         print(f"Service WARNING: Failed to fully send/update state for system prompt after mode change on active chat {chat_id}.")
            else:
                 # Case where the new mode is "Default" or has no prompt
                 print(f"Service: New mode '{new_mode}' has no system prompt. Nothing to send for active chat {chat_id}. Prompt flag remains reset (False).")
        else:
             # Case where the updated chat was not the active one
            print(f"Service: Chat {chat_id} (updated to mode '{new_mode}') is not the active chat. Prompt will be sent upon its next activation.")


    async def delete_chat(self, db: aiosqlite.Connection, chat_id: str):
        """Deletes a chat session from DB and cache."""
        print(f"Service: Attempting to delete chat {chat_id}")
        if chat_id not in self._cache:
             print(f"Service Warning: Chat {chat_id} not found in cache during deletion request.")
             raise HTTPException(status_code=404, detail="Chat session not found.")
        success_db = await self.repository.delete_chat(db, chat_id)
        if not success_db:
             print(f"Service Error: DB deletion failed for chat {chat_id} (it was present in cache).")
             raise HTTPException(status_code=500, detail=f"Failed to delete chat session from database for {chat_id}.")
        del self._cache[chat_id]
        print(f"Service: Chat {chat_id} removed from cache.")
        if self._active_chat_id == chat_id:
            self._active_chat_id = None
            print(f"Service: Deactivated chat {chat_id} because it was deleted.")

    def _extract_text_from_content(self, content) -> str:
        """Extract text from message content (string or list of blocks)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
            return "\n".join(text_parts)
        return ""

    def _extract_images_from_content(self, content) -> List[str]:
        """Extract base64 image data URIs from message content."""
        image_urls = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, ImageUrlBlock) and block.image_url.url.startswith("data:image"):
                    image_urls.append(block.image_url.url)
        return image_urls

    def _build_combined_prompt(self, messages: List[OpenAIMessage]) -> str:
        """
        Build a combined prompt from all messages in the request.
        
        When Roo Code (Kilo Code) sends requests, it includes system messages
        with its own tool use instructions, followed by user messages with the
        actual task. This method combines all messages into a single prompt
        that preserves the full context for Gemini.
        
        Message roles are mapped as follows:
        - system: Prepended as system instructions
        - user: Included as user content
        - assistant: Included as previous assistant responses (for context)
        """
        prompt_parts = []
        
        for msg in messages:
            text = self._extract_text_from_content(msg.content)
            if not text.strip():
                continue
                
            if msg.role == "system":
                prompt_parts.append(f"[System Instructions]\n{text}")
            elif msg.role == "assistant":
                prompt_parts.append(f"[Previous Assistant Response]\n{text}")
            elif msg.role == "user":
                prompt_parts.append(f"[User Message]\n{text}")
        
        return "\n\n".join(prompt_parts)

    async def handle_completion(self, db: aiosqlite.Connection, user_messages: List[OpenAIMessage]) -> ChatCompletionResponse:
        """
        Handles a chat completion request. Passes through ALL messages from the
        request (system, user, assistant) to Gemini as a combined prompt.
        
        When used with Roo Code (Kilo Code), the system messages contain the
        tool use instructions and mode-specific prompts that Roo Code generates.
        These are passed through transparently to Gemini.
        
        When used with the web UI (no system messages in request), falls back to
        the hardcoded prompt behavior via set_active_chat/update_chat_mode.
        
        Updates metadata in DB/cache afterwards.
        """
        if not self._active_chat_id:
            raise HTTPException(status_code=400, detail="No active chat session set. Use POST /v1/chats/active.")

        current_chat_id = self._active_chat_id
        print(f"Service: Handling completion for active chat: {current_chat_id}")

        # 1. Get session data from cache
        session_data = self._cache.get(current_chat_id)
        if not session_data:
            print(f"Service CRITICAL ERROR: Active chat ID '{current_chat_id}' is set but not found in cache!")
            self._active_chat_id = None
            raise HTTPException(status_code=404, detail=f"Active chat session '{current_chat_id}' state not found. Please set active chat again.")

        metadata = session_data.get("metadata")
        if metadata is None:
             print(f"Service CRITICAL ERROR: Metadata missing in cache for active chat {current_chat_id}!")
             raise HTTPException(status_code=500, detail="Internal error: Corrupted state for active chat.")

        # 2. Load Gemini ChatSession object
        try:
            chat_session = self.gemini_wrapper.load_chat_from_metadata(metadata=metadata)
            print(f"Service: Loaded Gemini ChatSession object for {current_chat_id}")
        except HTTPException as e: raise e
        except Exception as e:
             print(f"Service Error loading chat session from metadata: {e}")
             raise HTTPException(status_code=500, detail=f"Failed to load active chat session state: {e}")

        # 3. Check if request contains system messages (Roo Code / Kilo Code mode)
        has_system_messages = any(msg.role == "system" for msg in user_messages)
        
        # 4. Process images from the last user message
        last_user_message = next((msg for msg in reversed(user_messages) if msg.role == "user"), None)
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found in the request.")
        
        image_urls_to_process = self._extract_images_from_content(last_user_message.content)
        temp_file_paths = []
        
        try:
            for img_url in image_urls_to_process:
                try:
                    header, encoded = img_url.split(",", 1)
                    img_data = base64.b64decode(encoded)
                    mime_type = header.split(";")[0].split(":")[1] if ':' in header else 'application/octet-stream'
                    ext = mimetypes.guess_extension(mime_type) or ""
                    safe_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.heic', '.heif']
                    if ext.lower() in safe_extensions:
                        fd, temp_path = tempfile.mkstemp(suffix=ext)
                        os.write(fd, img_data)
                        os.close(fd)
                        temp_file_paths.append(temp_path)
                        print(f"Service: Saved image data URI ({mime_type}) to temp file: {temp_path}")
                    else:
                        print(f"Service Warning: Skipping image with potentially unsafe extension '{ext or 'unknown'}' from mime type '{mime_type}'")
                except Exception as img_e:
                    print(f"Service Error processing data URI: {img_e}. Skipping image.")
        except Exception as proc_e:
            self._cleanup_temp_files(temp_file_paths)
            raise HTTPException(status_code=400, detail=f"Error processing image content: {proc_e}")

        # 5. Build the prompt based on whether system messages are present
        if has_system_messages:
            # --- Roo Code / Kilo Code pass-through mode ---
            # Combine ALL messages (system + user + assistant) into a single prompt.
            # This passes through Roo Code's tool use instructions, mode prompts,
            # environment details, and user task transparently to Gemini.
            final_prompt_to_send = self._build_combined_prompt(user_messages)
            print(f"Service: Pass-through mode - combined {len(user_messages)} messages (including system) into prompt ({len(final_prompt_to_send)} chars)")
        else:
            # --- Legacy web UI mode ---
            # Only extract the last user message text, with optional mode switch detection
            user_message_text = self._extract_text_from_content(last_user_message.content).strip()
            
            if not user_message_text and not temp_file_paths:
                raise HTTPException(status_code=400, detail="No processable content found.")
            
            # Check for mode switch pattern (legacy web UI feature)
            mode_switch_match = re.search(r"\[switch_mode to '(.*?)' because:.*?\]", user_message_text, re.IGNORECASE | re.DOTALL)
            if mode_switch_match:
                extracted_mode = mode_switch_match.group(1).title()
                new_mode_prompt = MODE_PROMPT_TEXTS.get(extracted_mode)
                final_prompt_to_send = f"Now you are in {extracted_mode} mode. Use the following prompt:\n {new_mode_prompt}\n\n{user_message_text}"
            else:
                final_prompt_to_send = user_message_text
            
            print(f"Service: Legacy mode - sending user message only ({len(final_prompt_to_send)} chars)")

        if not final_prompt_to_send.strip() and not temp_file_paths:
            raise HTTPException(status_code=400, detail="No processable content found after building prompt.")

        # 6. Send to Gemini & Handle Response/State Update
        try:
            print(f"Service: Sending message to Gemini for chat {current_chat_id}...")
            api_response = await self.gemini_wrapper.send_message(
                chat_session=chat_session,
                prompt=final_prompt_to_send,
                files=temp_file_paths
            )
            response_text = getattr(api_response, 'text', "[No text in response]")
            # Unescape any backslash-escaped XML/markdown characters from Gemini
            response_text = self._unescape_response_text(response_text)
            print(f"Service: Response received from Gemini for chat {current_chat_id}.")

            # --- Update State Post-Gemini Call (Metadata ONLY) ---
            updated_metadata = chat_session.metadata
            print(f"Service: Updating metadata in DB for chat {current_chat_id}...")
            meta_update_ok = await self.repository.update_metadata(db, current_chat_id, updated_metadata)

            # Update cache metadata based on DB success
            if meta_update_ok:
                self._cache[current_chat_id]["metadata"] = updated_metadata
                # prompt_sent flag is NOT touched here
                print(f"Service: Metadata cache updated for chat {current_chat_id}.")
            else:
                 print(f"Service ERROR: Failed to update metadata in DB for {current_chat_id}. Cache may be stale.")

            # 7. Format Final API Response
            assistant_message = OpenAIMessage(role="assistant", content=response_text)
            choice = Choice(message=assistant_message)
            usage = Usage()
            openai_response = ChatCompletionResponse(
                model=GEMINI_MODEL_NAME, choices=[choice], usage=usage, chat_id=current_chat_id
            )
            return openai_response

        except HTTPException as e:
             print(f"Service Error (HTTPException) during completion for {current_chat_id}: {e.detail}")
             raise e
        except Exception as e:
             print(f"Service Error (General Exception) during completion for {current_chat_id}: {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail=f"Unexpected server error during chat completion: {e}")
        finally:
            # 8. Cleanup Temp Files
            self._cleanup_temp_files(temp_file_paths)

    def _unescape_response_text(self, text: str) -> str:
        """
        Unescape XML/markdown characters that Gemini may have escaped in tool use responses.
        
        Gemini sometimes returns responses with backslash-escaped characters like:
        - \\< instead of <
        - \\> instead of >
        - \\_ instead of _
        - \\* instead of *
        - \\[ instead of [
        - \\] instead of ]
        
        This method removes those escape sequences to produce valid XML tool calls
        that Kilo Code (Roo Code) can parse correctly.
        """
        if not text:
            return text
        
        # Unescape common markdown/XML special characters
        # Order matters - we want to handle the backslash escapes
        unescaped = text
        unescaped = unescaped.replace("\\<", "<")
        unescaped = unescaped.replace("\\>", ">")
        unescaped = unescaped.replace("\\_", "_")
        unescaped = unescaped.replace("\\*", "*")
        unescaped = unescaped.replace("\\[", "[")
        unescaped = unescaped.replace("\\]", "]")
        unescaped = unescaped.replace("\\`", "`")
        unescaped = unescaped.replace("\\#", "#")
        
        return unescaped

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Safely removes temporary files created for image uploads."""
        if file_paths:
            print(f"Service: Cleaning up {len(file_paths)} temporary image files...")
            for path in file_paths:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError as cleanup_e:
                    print(f"Service Error removing temp file '{path}': {cleanup_e}")
                except Exception as general_e:
                     print(f"Service Error during temp file '{path}' cleanup: {general_e}")