# Phase 3: Real-Time WebGL Rendering Architecture

You mentioned the ultimate goal is a **Real-Time ASL Avatar System**. 

Currently, your system is severely bottlenecked by its rendering pipeline. Right now, when a sentence is translated to SMPL-X parameters, the Flask backend uses `pyrender` to generate images frame-by-frame, uses `ffmpeg` to stitch them into an `.mp4` file, and sends that heavy video file over HTTP. This is not real-time; it takes seconds/minutes to generate a short clip.

To achieve true real-time generation and playback, we must transition to a **WebSocket + Three.js** architecture.

## 1. The Core Architecture Shift
Instead of doing the graphics rendering on the server, we will shift the rendering entirely to the user's GPU inside the browser.
*   **Backend (Python):** Generates the raw `[N, 182]` SMPL-X mathematical arrays (using your new Diffusion Model or FAISS). Instead of rendering video, it instantly streams this raw array to the frontend.
*   **Frontend (React/Three.js):** Receives the array and uses **WebGL** to instantly animate a 3D rigged character in the browser at 60 Frames Per Second.

## 2. Backend Modifications (Streaming Server)
I will modify `app.py` (or create a new service) to serve raw coordinate data.
- **New Endpoint:** `/api/stream_animation`
- **Output:** A compressed binary stream (or compact JSON) representing the `[N, 182]` parameters. Because we are sending raw floats instead of video frames, the payload size drops from Megabytes to a few Kilobytes, ensuring zero latency.

## 3. Frontend Modifications (Three.js Engine)
I will build a React Three Fiber (`@react-three/fiber`) rendering engine inside your Vite application.
- **The Rig:** We will need a base `.glb` or `.gltf` 3D model that matches the SMPL-X skeletal hierarchy (54 joints).
- **The Engine:** I will write the bone-mapping logic (`AnimationMixer` or direct quaternion application) that mathematically translates the 182 parameters (Root, Body, Hands, Jaw) directly onto the 3D mesh bones in real-time using `useFrame`.

## 4. User Interface Updates
I will update your `<YoutubeTab />` and `<SentencesTab />` components. Instead of a `<video>` tag with a loading spinner, there will be an interactive `<Canvas>` tag where the 3D avatar stands in a dynamic, modern lighting environment, waiting to execute signs in real-time as the data streams in.

---

## User Review Required / Open Questions

Before we execute this architectural overhaul:

1. **Do you have a SMPL-X Rigged Avatar?** To render in the browser, we need a 3D model file (e.g., `avatar.glb`) rigged to the standard SMPL-X skeleton. Do you have one, or would you like me to set up the engine to use a generic abstract mesh / bone-viewer as a placeholder?
2. **WebSockets vs REST:** For real-time, WebSockets are best, but standard REST returning the JSON array is easier to deploy initially. Are you okay with starting with REST + Three.js for V1 of the real-time viewer?
3. **Approval:** Do you approve of tearing out the backend `.mp4` video generation and replacing it with this browser-based WebGL engine?
