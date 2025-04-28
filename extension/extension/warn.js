// warn.js

// Extract the original URL from the query parameters
const urlParams = new URLSearchParams(window.location.search);
const originalUrl = urlParams.get('url');

// Redirect to the original URL when the button is clicked
const continueBtn = document.getElementById('continueBtn');
if (originalUrl) {
    continueBtn.addEventListener('click', () => {
        // window.location.href = originalUrl;

        // Send a message to the background script to allow this URL
        chrome.runtime.sendMessage({ action: 'allowUrl', url: originalUrl }, (response) => {
            // Redirect to the original URL after adding it to the allowlist
            window.location.href = originalUrl;
        });
    });
} else {
    // If no URL is provided, disable the button
    continueBtn.disabled = true;
    continueBtn.textContent = 'No URL Provided';
}