import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import time
import threading

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = 'E:\\WORKSPACES\\Qr\\Libraries\\Open3D\\examples\\test_data'
RGB_DIR = os.path.join(DATA_DIR, 'RGBD/color')
DEPTH_DIR = os.path.join(DATA_DIR, 'RGBD/depth')


# ------------------------------- Main content ------------------------------- #
def rescale_greyscale(img):
    data = np.asarray(img)
    assert (len(data.shape) == 2)
    data_float = data.astype(np.float64)
    max_val = data_float.max()
    data_float *= 255. / max_val
    data8 = data_float.astype(np.uint8)
    return o3d.geometry.Image(data8)


class VideoWindow:
    def __init__(self):
        self.rgb_images = []
        for f in os.listdir(RGB_DIR):
            if f.endswith('.jpg') or f.endswith('.png'):
                img = o3d.io.read_image(os.path.join(RGB_DIR, f))
                self.rgb_images.append(img)
        self.depth_images = []
        for f in os.listdir(DEPTH_DIR):
            if f.endswith('.jpg') or f.endswith('.png'):
                img = o3d.io.read_image(os.path.join(DEPTH_DIR, f))
                img = rescale_greyscale(img)
                self.depth_images.append(img)
        assert (len(self.rgb_images) == len(self.depth_images))

        self.window = gui.Application.instance.create_window(
            'Open3D - Video Example', 1000, 500)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        lit = rendering.Material()
        lit.shader = "defaultLit"
        tet = o3d.geometry.TriangleMesh.create_tetrahedron()
        tet.compute_vertex_normals()
        tet.paint_uniform_color([0.5, 0.75, 1.0])

        self.widget3d.scene.add_geometry('tetrahedron', tet, lit)
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        self.widget3d.scene.show_axes(True)

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        # Color image
        self.panel.add_child(gui.Label('Color image'))
        self.rgb_widget = gui.ImageWidget(self.rgb_images[0])
        self.panel.add_child(self.rgb_widget)
        # Depth image
        self.panel.add_child(gui.Label('Depth image(normalized)'))
        self.depth_widget = gui.ImageWidget(self.depth_images[0])
        self.panel.add_child(self.depth_widget)
        self.window.add_child(self.panel)

        self.is_done = False
        threading.Thread(target=self._update_thread).start()

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size
        self.widget3d.frame = gui.Rect(
            content_rect.x, content_rect.y,
            content_rect.width - panel_width,
            content_rect.height)
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(), content_rect.y,
            panel_width, content_rect.height)

    def _on_close(self):
        self.is_done = True
        return True

    def _update_thread(self):
        idx = 0
        while not self.is_done:
            time.sleep(0.1)

            # Get the next frame
            rgb_frame = self.rgb_images[idx]
            depth_frame = self.depth_images[idx]
            idx += 1
            if idx >= len(self.rgb_images):
                idx = 0

            # Update the images
            def update():
                self.rgb_widget.update_image(rgb_frame)
                self.depth_widget.update_image(depth_frame)
                self.widget3d.scene.set_background([1, 1, 1, 1], rgb_frame)

            if not self.is_done:
                gui.Application.instance.post_to_main_thread(
                    self.window, update)

# ---------------------------------------------------------------------------- #


# ---------------------------------- Testing --------------------------------- #
def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    window = VideoWindow()

    app.run()


if __name__ == '__main__':
    main()
# ---------------------------------------------------------------------------- #
