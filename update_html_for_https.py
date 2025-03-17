#!/usr/bin/env python3
"""
Script to update HTML files to use relative URLs and protocol-relative URLs
to ensure compatibility with both HTTP and HTTPS.
"""

import os
import re
import glob

def update_html_file(file_path):
    """Update a single HTML file to use relative or protocol-relative URLs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count original occurrences of http:// URLs
    http_count = len(re.findall(r'http://', content))
    
    # Replace hardcoded http:// URLs with protocol-relative URLs (//example.com)
    # This will use the same protocol as the page (http or https)
    content = re.sub(r'http://([\w\d\.-]+)/', r'//\1/', content)
    
    # Replace hardcoded localhost URLs with relative URLs
    content = re.sub(r'(http:|https:)?//localhost:5000/', r'/', content)
    
    # Update WebSocket connections to use secure WebSockets when on HTTPS
    content = re.sub(r'new WebSocket\("ws:', r'new WebSocket(window.location.protocol === "https:" ? "wss:" : "ws:', content)
    
    # Add meta tag to ensure HTTPS if not already present
    if '<meta http-equiv="Content-Security-Policy"' not in content:
        meta_tag = '<meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">'
        content = content.replace('</head>', f'    {meta_tag}\n</head>', 1)
    
    # Add JavaScript to handle protocol switching for APIs
    if 'function getBaseUrl()' not in content:
        js_code = """
    <script>
        // Helper function to get the base URL with the correct protocol
        function getBaseUrl() {
            return window.location.protocol + '//' + window.location.host;
        }
        
        // Update all fetch/ajax calls to use the base URL
        document.addEventListener('DOMContentLoaded', function() {
            // Replace any remaining hardcoded URLs in onclick attributes
            document.querySelectorAll('[onclick*="http://"]').forEach(function(el) {
                el.setAttribute('onclick', el.getAttribute('onclick').replace('http://', '//'));
            });
        });
    </script>
"""
        content = content.replace('</head>', f'{js_code}\n</head>', 1)
    
    # Replace any fetch or AJAX calls with getBaseUrl()
    content = re.sub(r'fetch\([\'"]http://[\w\d\.-]+:5000/([^\'"]+)[\'"]', r'fetch(getBaseUrl() + "/\1"', content)
    content = re.sub(r'url: [\'"]http://[\w\d\.-]+:5000/([^\'"]+)[\'"]', r'url: getBaseUrl() + "/\1"', content)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Return the number of http:// occurrences found
    return http_count

def main():
    """Update all HTML files in the templates directory."""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    html_files = glob.glob(os.path.join(template_dir, '*.html'))
    
    if not html_files:
        print("No HTML files found in the templates directory.")
        return
    
    total_files = len(html_files)
    total_http_urls = 0
    
    print(f"Found {total_files} HTML files to process.")
    
    for file_path in html_files:
        file_name = os.path.basename(file_path)
        http_count = update_html_file(file_path)
        total_http_urls += http_count
        print(f"Updated {file_name} - Found {http_count} HTTP URLs")
    
    print(f"\nCompleted! Updated {total_files} files, replaced {total_http_urls} HTTP URLs.")
    print("Your application should now work correctly with both HTTP and HTTPS.")

if __name__ == "__main__":
    main() 