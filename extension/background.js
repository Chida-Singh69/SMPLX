chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    enabled: true,
    gender: 'neutral',
    backendUrl: 'http://127.0.0.1:5000',
    overlayPosition: 'top-right'
  });
});
