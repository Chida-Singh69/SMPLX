console.log("SMPL-X ASL Extension Active");

// Content script logic for text selection and overlay trigger
document.addEventListener("mouseup", (event) => {
  const selection = window.getSelection().toString().trim();
  if (selection.length > 0) {
    console.log("Selected text:", selection);
    // Future: Trigger overlay or send to backend
  }
});
