document.addEventListener('DOMContentLoaded', () => {
  const enableToggleEl = document.getElementById('enableToggle');
  const genderSelectEl = document.getElementById('genderSelect');
  const saveBtnEl = document.getElementById('saveBtn');
  const toggleBtnEl = document.getElementById('toggleBtn');

  function setToggleBtnLabel(enabled) {
    toggleBtnEl.textContent = enabled ? 'Disable Overlay' : 'Enable Overlay';
  }

  function notifyActiveTab(enabled) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs && tabs[0];
      if (!tab || !tab.id) return;
      chrome.tabs.sendMessage(tab.id, { type: 'sv-toggle-overlay', enabled });
    });
  }

  chrome.storage.local.get(['enabled', 'gender'], (result) => {
    const enabled = result.enabled !== false;
    enableToggleEl.checked = enabled;
    setToggleBtnLabel(enabled);
    if (result.gender) {
      genderSelectEl.value = result.gender;
    }
  });

  saveBtnEl.addEventListener('click', () => {
    const enabled = enableToggleEl.checked;
    const gender = genderSelectEl.value;
    
    chrome.storage.local.set({ enabled, gender }, () => {
      saveBtnEl.textContent = 'Saved!';
      setTimeout(() => saveBtnEl.textContent = 'Save Settings', 1500);
      setToggleBtnLabel(enabled);
      notifyActiveTab(enabled);
    });
  });

  toggleBtnEl.addEventListener('click', () => {
    chrome.storage.local.get(['enabled'], (result) => {
      const currentlyEnabled = result.enabled !== false;
      const enabled = !currentlyEnabled;
      chrome.storage.local.set({ enabled }, () => {
        enableToggleEl.checked = enabled;
        setToggleBtnLabel(enabled);
        notifyActiveTab(enabled);
      });
    });
  });
});
