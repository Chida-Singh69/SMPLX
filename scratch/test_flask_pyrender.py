import threading
from flask import Flask, Response
import sys
sys.path.append('backend/core')
from word_to_smplx import PYRENDER_AVAILABLE
print("Pyrender available:", PYRENDER_AVAILABLE)

app = Flask(__name__)

@app.route('/test')
def test():
    def generate():
        import pyrender
        import numpy as np
        renderer = pyrender.OffscreenRenderer(640, 480)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])
        color, depth = renderer.render(scene)
        yield "data: done\n\n"
        # renderer.delete()
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5005, debug=False, use_reloader=False)
