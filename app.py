import streamlit as st
import os
import zipfile
import tempfile
import base64
import shutil
from bs4 import BeautifulSoup
import io
import google.generativeai as genai
from pathlib import Path
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Web Page Language Converter",
    page_icon="🌐",
    layout="wide"
)

# App title and description
st.title("🌐 Web Page Language Converter")
st.markdown("""
This app converts web pages from English to your desired language using Google's Gemini AI.
Upload either a single HTML file or a ZIP file containing multiple web files.
""")

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    api_key = st.secrets["API_KEY"]
    target_language = st.selectbox(
        "Select Target Language",
        ["Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Russian", "Arabic", "Portuguese", "Italian"]
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses Google's Gemini AI to translate web content.
    - Preserves HTML structure
    - Works with single files or ZIP archives
    - Maintains CSS and JS references
    """)


def configure_genai_client(api_key):
    """Configure the Gemini AI client with the provided API key."""
    if not api_key:
        st.error("Please enter a valid Gemini API key.")
        return False

    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini AI: {str(e)}")
        return False


def translate_text_with_genai(text, target_language):
    """Use Gemini AI to translate the text to the target language.
    This function sends the entire HTML file for translation."""
    if not text.strip():
        return text

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

        # Prompt engineering to ensure proper translation of entire HTML file
        prompt = f"""
        Translate this entire HTML document from English to {target_language}.
        IMPORTANT INSTRUCTIONS:
        1. Preserve ALL HTML tags, attributes, and structure EXACTLY as they are
        2. Only translate the natural language text content that would be visible to users
        3. DO NOT modify any:
           - HTML tags or attributes
           - JavaScript code
           - CSS rules
           - URL paths
           - ID or class names
           - Special characters or entities

        HTML document to translate:
        {text}
        """

        # Set a higher max output tokens to handle large HTML files
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.2
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

        # Clean the response to ensure we get only HTML content
        result = response.text

        # Handle if Gemini adds any markdown code blocks
        if result.startswith("```html") and result.endswith("```"):
            result = result[7:-3]  # Remove the markdown code block markers

        return result
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text


def process_html_file(file_content, target_language):
    """Process a single HTML file to translate its content.
    Sends the entire HTML file to Gemini for translation."""
    try:
        # Send the entire HTML content to Gemini for translation
        translated_content = translate_text_with_genai(file_content, target_language)
        return translated_content
    except Exception as e:
        st.error(f"Error processing HTML: {str(e)}")
        return file_content


def handle_zip_file(zip_content, target_language):
    """Process a zip file containing web files and translate HTML content.
    Maintains the exact folder structure of the original zip file."""
    # Create temporary directories to work with
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Extract zip file to input directory
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
            zip_ref.extractall(input_dir)

        # Track progress
        html_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.html') or file.lower().endswith('.htm'):
                    html_files.append(os.path.join(root, file))

        total_files = len(html_files)
        if total_files > 0:
            progress_bar = st.progress(0)

        # Process each file in the zip, maintaining folder structure
        processed_count = 0
        for root, dirs, files in os.walk(input_dir):
            # First create all directories in the output dir
            for dir_name in dirs:
                input_dir_path = os.path.join(root, dir_name)
                rel_dir_path = os.path.relpath(input_dir_path, input_dir)
                output_dir_path = os.path.join(output_dir, rel_dir_path)
                os.makedirs(output_dir_path, exist_ok=True)

            # Then process all files
            for file in files:
                input_file_path = os.path.join(root, file)
                # Create corresponding path in output directory
                rel_path = os.path.relpath(input_file_path, input_dir)
                output_file_path = os.path.join(output_dir, rel_path)

                # Create directory structure if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # If HTML file, send entire file for translation
                if file.lower().endswith('.html') or file.lower().endswith('.htm'):
                    try:
                        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            html_content = f.read()

                        # Show which file is being processed
                        st.text(f"Processing: {rel_path}")

                        # Send entire HTML content to Gemini
                        translated_html = process_html_file(html_content, target_language)

                        with open(output_file_path, 'w', encoding='utf-8') as f:
                            f.write(translated_html)

                        # Update progress
                        processed_count += 1
                        if total_files > 0:
                            progress_bar.progress(processed_count / total_files)

                    except Exception as e:
                        st.warning(f"Error processing {rel_path}: {str(e)}")
                        # Copy original file if translation fails
                        shutil.copy2(input_file_path, output_file_path)
                else:
                    # Copy non-HTML files without modification
                    shutil.copy2(input_file_path, output_file_path)

        # Create a zip file from the output directory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zf.write(file_path, arcname)

        memory_file.seek(0)
        return memory_file.getvalue()


def create_download_link(file_content, file_name):
    """Create a download link for the processed file."""
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download Translated File</a>'
    return href


# Main application logic
uploaded_file = st.file_uploader("Upload HTML or ZIP file", type=["html", "htm", "zip", "rar"])

if uploaded_file and api_key:
    if not configure_genai_client(api_key):
        st.stop()

    # Check file type
    file_content = uploaded_file.read()
    file_extension = uploaded_file.name.split('.')[-1].lower()

    with st.spinner(f"Translating to {target_language}... Please wait"):
        if file_extension in ['html', 'htm']:
            # Process single HTML file
            try:
                st.info(f"Translating HTML file to {target_language}...")

                translated_content = process_html_file(file_content.decode('utf-8', errors='ignore'), target_language)
                download_content = translated_content.encode('utf-8')
                output_filename = f"translated_{uploaded_file.name}"

                # Show success message
                st.success("Translation complete!")

                # Show preview with expandable section
                with st.expander("Preview of Translated Content"):
                    st.code(translated_content[:1500] + "..." if len(translated_content) > 1500 else translated_content,
                            language="html")

                # Provide download link
                st.markdown("### Download Your Translated File:")
                st.markdown(create_download_link(download_content, output_filename), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing HTML file: {str(e)}")

        elif file_extension in ['zip', 'rar']:
            # Process archive file
            try:
                st.info(f"Processing archive file and translating HTML content to {target_language}...")
                st.warning("This may take some time depending on the number and size of HTML files.")

                # Process the archive
                if file_extension == 'zip':
                    processed_zip = handle_zip_file(file_content, target_language)
                else:  # RAR files get handled as ZIP files after extraction
                    st.error("RAR format detected. Please convert to ZIP format first.")
                    st.stop()

                output_filename = f"translated_{uploaded_file.name}"

                # Provide download link
                st.success("Archive processing complete!")
                st.markdown("### Download Your Translated Archive:")
                st.markdown(create_download_link(processed_zip, output_filename), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing archive file: {str(e)}")
                st.error("If you're using a RAR file, please convert it to ZIP format first.")
        else:
            st.error(f"Unsupported file format: {file_extension}")

# Instructions for usage
if not uploaded_file:
    st.info("👆 Please upload an HTML file or a ZIP file containing web content to translate.")

# Additional information
st.markdown("---")
st.markdown("""
### How to Use

1. Enter your Gemini API key in the sidebar
2. Select your target language
3. Upload an HTML file or ZIP archive
4. Wait for processing to complete
5. Download your translated file

For best results with complex websites, ensure all resources (CSS, JS, images) are included in your ZIP file.
""")
