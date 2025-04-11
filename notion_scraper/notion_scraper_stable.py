import os
import json
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# Base URL of the Notion wiki
BASE_URL = "https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4"
# Directory to save the JSON files
SAVE_DIR = "../prosperity_wiki"
# Directory to save code files
CODE_DIR = "../prosperity_wiki/code"

def save_json(data, folder, filename):
    """Save data to a JSON file."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def determine_category(title, content_blocks):
    """Determine which category the page belongs to based on title and content."""
    title_lower = title.lower()
    
    # Check for round-related pages
    if re.search(r'round \d|tutorial round', title_lower):
        return "rounds"
    
    # Check for e-learning related content
    e_learning_keywords = ["glossary", "resources", "algorithm", "programming", "python", "learning"]
    if any(keyword in title_lower for keyword in e_learning_keywords):
        return "e-learning_center"
    
    # Default to about_prosperity for general information
    return "about_prosperity"

def save_code_file(code_content, language, page_title, code_id):
    """Save code to a separate file and return the file path."""
    # Create sanitized file name base
    sanitized_title = page_title.lower().replace(" ", "_").replace("/", "_")
    
    # Determine file extension based on language
    extension = ".py" if language.lower() == "python" else ".txt"
    
    # Create filename
    filename = f"{sanitized_title}_{code_id}{extension}"
    
    # Ensure code directory exists
    os.makedirs(CODE_DIR, exist_ok=True)
    
    # Save the code file
    file_path = os.path.join(CODE_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_content)
    
    # Return the relative path from the project root
    return os.path.join("code", filename)

def extract_content(soup, page_title):
    """Extract content blocks from the Notion page in their natural order."""
    blocks = []
    code_block_counter = 1  # Counter for code blocks
    
    # First get the page title (h1) which is special
    title_div = soup.select_one(".notion-page-block h1")
    if title_div:
        blocks.append({
            "type": "h1",
            "content": title_div.get_text().strip()
        })
    
    # Get the main content container
    content_container = soup.select_one(".notion-page-content")
    if not content_container:
        return blocks  # Early return if no content found
    
    # Track consecutive list elements to combine them later
    current_list_type = None
    current_list_items = []
    
    # Process each child element in order
    for element in content_container.find_all(class_=True, recursive=False):
        class_name = element.get("class", [])
        class_name_str = " ".join(class_name)
        
        # Determine element type based on class
        is_bulleted_list = "notion-bulleted-list" in class_name or "notion-bulleted_list" in class_name_str
        is_numbered_list = "notion-numbered-list" in class_name or "notion-numbered_list" in class_name_str
        
        # Check if this is any kind of list element
        is_list_element = is_bulleted_list or is_numbered_list
        list_style = "bulleted" if is_bulleted_list else "numbered" if is_numbered_list else None
        
        # If encountering a new type of element or different list style, 
        # flush the current list if we've been building one
        if (not is_list_element or list_style != current_list_type) and current_list_items:
            blocks.append({
                "type": "list",
                "style": current_list_type,
                "items": current_list_items
            })
            current_list_items = []
            current_list_type = None
        
        # Process based on element type
        if is_list_element:
            # This is a list element - extract and add to current items
            extracted_items = extract_list_items(element, list_style)
            if extracted_items:
                current_list_type = list_style
                current_list_items.extend(extracted_items)
                
        elif "notion-header-block" in class_name:
            # Handle headings
            style = element.get('style', '')
            if 'font-size: 1.5em' in style or 'font-weight: 700' in style:
                heading_type = "h2"
            elif 'font-size: 1.25em' in style or 'font-weight: 600' in style:
                heading_type = "h3"
            elif 'font-size: 1em' in style or 'font-weight: 500' in style:
                heading_type = "h4"
            else:
                heading_type = "h3"
            
            heading_text = element.get_text().strip()
            if heading_text:
                blocks.append({
                    "type": heading_type,
                    "content": heading_text
                })
                
        elif "notion-text-block" in class_name:
            # Handle paragraphs
            text = element.get_text().strip()
            if text:
                blocks.append({
                    "type": "p",
                    "content": text
                })
                
        elif "notion-code-block" in class_name:
            # Handle code blocks
            code = element.get_text().strip()
            if code:
                # Remove "PythonCopy" prefix if present
                code = re.sub(r'^PythonCopy', '', code).strip()
                
                # Generate a unique ID for this code block
                code_id = f"code_{code_block_counter}"
                code_block_counter += 1
                
                # Determine language (default to Python if not specified)
                language = "python"  # Default language
                
                # Try to detect language from class or content
                for cls in class_name:
                    lang_match = re.search(r'language-(\w+)', cls)
                    if lang_match:
                        language = lang_match.group(1)
                        break
                
                # Save the code to a separate file
                file_path = save_code_file(code, language, page_title, code_id)
                
                # Add reference to the code file in the JSON
                blocks.append({
                    "type": "code",
                    "language": language,
                    "code_id": code_id,
                    "file_path": file_path,
                    "preview": code[:50] + ("..." if len(code) > 50 else "")  # Short preview
                })
    
    # Don't forget to add any remaining list items
    if current_list_items:
        blocks.append({
            "type": "list",
            "style": current_list_type,
            "items": current_list_items
        })
    
    return blocks

def extract_list_items(list_element, list_style):
    """Extract items from a list element with proper nesting level."""
    items = []
    
    # Different selectors for list items based on Notion's structure
    list_item_selectors = [
        ".notion-list-item", 
        "li",  # Standard list items
        "[data-block-id]"  # Block-based items
    ]
    
    # Try different selectors to find list items
    list_items_elements = []
    for selector in list_item_selectors:
        found_items = list_element.select(selector)
        if found_items:
            list_items_elements = found_items
            break
    
    # If we couldn't find list items with specific selectors, try direct children
    if not list_items_elements:
        list_items_elements = list_element.find_all(recursive=False)
    
    for item in list_items_elements:
        # Extract text content
        text = item.get_text().strip()
        if not text:
            continue
            
        # Determine nesting level based on indentation or class
        level = 0
        
        # Check for indentation in style
        style = item.get('style', '')
        if 'padding-left' in style or 'margin-left' in style:
            # Extract pixels and estimate level
            pixels = re.search(r'(?:padding|margin)-left:\s*(\d+)px', style)
            if pixels:
                # Typically, each indent level is around 24-30px
                level = int(pixels.group(1)) // 24
                
        # Alternatively check for specific classes that indicate nesting
        if "notion-item-indent" in " ".join(item.get('class', [])):
            for cls in item.get('class', []):
                indent_match = re.search(r'notion-item-indent-(\d+)', cls)
                if indent_match:
                    level = int(indent_match.group(1))
        
        items.append({
            "content": text,
            "level": level
        })
    
    return items

def scrape_notion_wiki():
    """Main function to scrape the Notion wiki and save content as JSON."""
    with sync_playwright() as p:
        # Use a more complete browser configuration
        browser = p.chromium.launch(
            headless=False,  # Use headed mode for debugging
            args=['--disable-web-security', '--disable-features=IsolateOrigins', '--disable-site-isolation-trials']
        )
        
        # Create a more complete browser context
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
        
        page = context.new_page()
        print(f"Accessing: {BASE_URL}")
        
        try:
            # Change the wait_until strategy to 'domcontentloaded' instead of 'load'
            page.goto(BASE_URL, timeout=60000, wait_until='domcontentloaded')
            print("Page loaded successfully")
            
            # Take a screenshot to see what's loaded
            print("Taking an initial screenshot...")
            os.makedirs("debug", exist_ok=True)
            page.screenshot(path="debug/initial_load.png")
        except Exception as e:
            print(f"Error loading page: {e}")
            # Save the error information for debugging
            with open("debug/error_log.txt", "w") as f:
                f.write(f"Error loading page: {e}")
            # Try to take screenshot of whatever did load
            try:
                page.screenshot(path="debug/error_state.png")
            except:
                pass
        
        # Wait for content to load and interact with the page
        try:
            # Wait longer for the page to load completely
            print("Waiting for page content to load...")
            page.wait_for_timeout(15000)
            
            # Try to find and interact with page elements
            print("Scrolling page to trigger content loading...")
            # Scroll down to trigger lazy loading
            page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight / 2);
                setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 2000);
            """)
            page.wait_for_timeout(5000)
            
            # Try to click any "Show more" or expand buttons
            try:
                expand_buttons = page.query_selector_all('button:has-text("Show more")')
                for button in expand_buttons:
                    button.click()
                    page.wait_for_timeout(1000)
            except Exception as e:
                print(f"No expand buttons found: {e}")
                
            # Take another screenshot after interactions
            page.screenshot(path="debug/after_interaction.png")
        except Exception as e:
            print(f"Error during page interaction: {e}")
        
        # Try multiple selector strategies to find links
        internal_links = set()
        
        # Strategy 1: Standard link selector
        try:
            print("Looking for links with standard selector...")
            links = page.query_selector_all('a[href*="notion.site"]')
            for link in links:
                href = link.get_attribute('href')
                if href and "notion.site" in href and "Prosperity-3" in href:
                    internal_links.add(href.split('?')[0])  # Remove URL parameters
        except Exception as e:
            print(f"Error with standard link selector: {e}")
            
        # Strategy 2: Notion specific selectors
        try:
            print("Looking for links with Notion-specific selectors...")
            notion_links = page.query_selector_all('.notion-link-token a, .notion-selectable a, .notion-page-block a')
            for link in notion_links:
                href = link.get_attribute('href')
                if href and "notion.site" in href:
                    internal_links.add(href.split('?')[0])
        except Exception as e:
            print(f"Error with Notion-specific selector: {e}")
            
        # Strategy 3: JavaScript execution to extract links
        try:
            print("Extracting links using JavaScript...")
            links_from_js = page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a'));
                    return links
                        .filter(a => a.href && a.href.includes('notion.site'))
                        .map(a => a.href.split('?')[0]);
                }
            """)
            for link in links_from_js:
                if "Prosperity-3" in link or "prosperity" in link.lower():
                    internal_links.add(link)
        except Exception as e:
            print(f"Error extracting links with JavaScript: {e}")
        
        print(f"Found {len(internal_links)} internal pages.")
        
        # Debug: If still no links, print the page content
        if len(internal_links) == 0:
            print("No links found. Saving page HTML for debugging...")
            with open("notion_debug.html", "w", encoding="utf-8") as f:
                f.write(page.content())
            print("Debugging HTML saved to notion_debug.html")

        index = []

        for link in internal_links:
            print(f"Scraping: {link}")
            page.goto(link)
            page.wait_for_timeout(8000)  # Wait longer 
            soup = BeautifulSoup(page.content(), "html.parser")
            
            title_tag = soup.find("title")
            title = title_tag.text.strip() if title_tag else "Untitled"
            
            # Pass the page title to extract_content for code file naming
            content_blocks = extract_content(soup, title)

            # Generate a filename based on the page title
            file_name = title.lower().replace(" ", "_").replace("/", "_") + ".json"
            
            # Determine which category this page belongs to
            category = determine_category(title, content_blocks)
            # Create the full path for the file
            folder = os.path.join(SAVE_DIR, category)
            save_json(content_blocks, folder, file_name)

            index.append({
                "title": title,
                "path": f"{category}/{file_name}",
                "source_url": link
            })

        # Save the index of all pages
        save_json(index, SAVE_DIR, "index.json")

        browser.close()
        print("Scraping and conversion completed.")

if __name__ == "__main__":
    scrape_notion_wiki()