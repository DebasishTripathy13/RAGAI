import logging
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config import REQUEST_TIMEOUT, runtime_config

def get_sitemap_urls(base_url):
    """
    Find and process XML sitemaps to extract URLs.
    1. Tries the standard /sitemap.xml location
    2. If not found, checks robots.txt for Sitemap entries
    3. Processes sitemap XML to extract URLs
    """
    sitemap_urls = []
    sitemap_locations = []
    
    # Normalize base URL
    parsed_url = urlparse(base_url)
    base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Try standard sitemap location
    try:
        standard_sitemap_url = f"{base_domain}/sitemap.xml"
        response = requests.get(standard_sitemap_url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200 and "xml" in response.headers.get("content-type", ""):
            sitemap_locations.append(standard_sitemap_url)
    except requests.exceptions.RequestException:
        pass
    
    # If no standard sitemap, check robots.txt
    if not sitemap_locations:
        try:
            robots_url = f"{base_domain}/robots.txt"
            response = requests.get(robots_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        sitemap_locations.append(sitemap_url)
        except requests.exceptions.RequestException:
            pass

    # Process each sitemap location
    for sitemap_url in sitemap_locations:
        try:
            response = requests.get(sitemap_url, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                continue

            # Handle both standard sitemaps and sitemap indexes
            try:
                root = ET.fromstring(response.content)
                
                # Check if this is a sitemap index
                is_sitemap_index = False
                namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                
                # Check for sitemap tags (indicates sitemap index)
                sitemaps = root.findall(".//sm:sitemap", namespaces) or root.findall(".//sitemap")
                if sitemaps:
                    is_sitemap_index = True
                    # Process sitemap index - get URLs of actual sitemaps
                    for sitemap in sitemaps:
                        loc = sitemap.find(".//sm:loc", namespaces) or sitemap.find(".//loc")
                        if loc is not None and loc.text:
                            # Don't process too many sitemaps from the index
                            if len(sitemap_locations) >= 5:  # Limit to 5 sitemaps
                                break
                            sitemap_locations.append(loc.text.strip())
                else:
                    # This is a standard sitemap - extract page URLs
                    urls = root.findall(".//sm:url/sm:loc", namespaces) or root.findall(".//url/loc")
                    for url in urls:
                        if url.text and url.text.strip():
                            sitemap_urls.append(url.text.strip())
                            if len(sitemap_urls) >= runtime_config.max_sitemap_urls:
                                break
                    
            except ET.ParseError:
                logging.warning(f"Could not parse XML from {sitemap_url}")
                # Try parsing as HTML (sometimes sitemaps are HTML)
                try:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if "http" in href and not href.endswith(('.jpg', '.png', '.css', '.js')):
                            sitemap_urls.append(href)
                            if len(sitemap_urls) >= runtime_config.max_sitemap_urls:
                                break
                except Exception:
                    pass
                
        except requests.exceptions.RequestException as e:
            logging.warning(f"Error fetching sitemap {sitemap_url}: {e}")
    
    # Ensure we don't exceed the maximum number of URLs to process
    return sitemap_urls[:runtime_config.max_sitemap_urls]