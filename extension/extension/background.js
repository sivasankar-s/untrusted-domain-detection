const alertPage = chrome.extension.getURL("block.html");
const warnPage = chrome.extension.getURL("warn.html");

async function checkDomain(url) {
    try {
        // Extract the domain from the URL
        const domain = new URL(url).hostname;

        const API_URL = "http://127.0.0.1:8000/predict";

        // Call the FastAPI endpoint
        const response = await fetch(`${API_URL}`, {
            method: "POST", 
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ domain: domain })
        });

        if (!response.ok) throw new Error("API Error");

        const result = await response.json();

        console.log("API Response:", result);

        return !result.isUntrusted;
    } catch (error) {
        console.error("Error checking domain:", error);
        return true;
    }
}


function ouralgorithm(requestDetails) {
	return new Promise(function(resolve){
		let isTrue = checkDomain(requestDetails.url);
        resolve(isTrue);
    })
}

function get_scheme(original_url){
	return original_url.split(":")[0];
}

// Temporary allowlist to store URLs the user has approved
const allowedUrls = new Set();

async function redirect(requestDetails){
	var redirectDest = alertPage;
    var redirectDest2 = `${warnPage}?url=${encodeURIComponent(requestDetails.url)}`;
    var scheme = get_scheme(requestDetails.url);
    console.log(scheme)
	

    var final_result = null
    await ouralgorithm(requestDetails).then(function(result){
        final_result = result
        console.log(result)
    })
    console.log(final_result)
    httpwarnflag = true;

    // Warn if the URL is HTTP and not in the temporary allowlist
    if (httpwarnflag && scheme == 'http' && !allowedUrls.has(requestDetails.url)) {
        console.log('warn HTTP')
        chrome.tabs.update(requestDetails.tabId, { url: redirectDest2 });
    }

    // Block the URL if it is untrusted
    if (final_result == false) {
        console.log(redirectDest)
        httpwarnflag = false;
        chrome.tabs.update(requestDetails.tabId, { url: redirectDest });
    }

    
}

// Listen for messages from the warning page
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'allowUrl') {
        const url = message.url;
        console.log(`Adding ${url} to the allowlist.`);
        allowedUrls.add(url); // Add the URL to the allowlist
    }
});


chrome.webRequest.onBeforeRequest.addListener(
    redirect,
    {urls:["*://*/*"],types:["main_frame"]},
    ["blocking"]
  );