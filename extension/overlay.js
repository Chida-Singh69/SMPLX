// Logic for the video overlay UI
function createOverlay() {
  const container = document.createElement("div");
  container.id = "smplx-asl-overlay";
  container.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 320px;
    height: 240px;
    background: #000;
    border-radius: 8px;
    z-index: 9999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  `;
  document.body.appendChild(container);
}
