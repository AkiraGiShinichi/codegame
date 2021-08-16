import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import time
import threading

from realsense_capture import RealsenseCapture, convert_depth_frame_to_points, post_process_depth_frame


def rescale_greyscale(img):
    data = np.asarray(img)
    assert (len(data.shape) == 2)
    data_float = data.astype(np.float64)
    max_val = data_float.max()
    data_float *= 255. / max_val
    data8 = data_float.astype(np.uint8)
    return o3d.geometry.Image(data8)


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def to_o3d_image(image):
    return o3d.geometry.Image(image)


def to_o3d_cloud(xyz, rgb=None, rgb_norm=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)
    if rgb_norm is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb_norm)
    return pcd


def to_pick_out(arrays, conditions):
    assert isinstance(arrays, tuple), 'Not be tuple of arrays'
    return [array[conditions] for array in arrays]


class VideoWindow:
    def __init__(self):
        self.window = gui.Application.instance.create_window(
            'Quadrep GUI', 1000, 500)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.enable_scene_caching(True)
        self.window.add_child(self.widget3d)

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        self.rgb_widget = gui.ImageWidget()
        self.depth_widget = gui.ImageWidget()

        self.panel.add_child(gui.Label('Color Image'))
        self.panel.add_child(self.rgb_widget)
        self.panel.add_child(gui.Label('Depth Image'))
        self.panel.add_child(self.depth_widget)

        self.window.add_child(self.panel)

        self.capture = RealsenseCapture(
            depth_size=(640, 480),
            # color_size=(960, 540),
            color_size=(640, 480),
            fps=30)  # L515
        self.capture.enable_device()
        self.capture.warm_up(30)

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
        while not self.is_done:
            time.sleep(0.1)
            if self.capture.isOpened():
                # Capture image
                status, images = self.capture.read(
                    return_depth=True, depth_filter=post_process_depth_frame)
                # Display image
                if status:
                    color_image, depth_image = images

                    x, y, z = convert_depth_frame_to_points(
                        depth_image, self.capture.get_intrinsics())
                    r = color_image[:, :, 0].flatten()
                    g = color_image[:, :, 1].flatten()
                    b = color_image[:, :, 2].flatten()

                    r, g, b, x, y, z = to_pick_out(
                        arrays=(r, g, b, x, y, z),
                        conditions=np.nonzero(z)[0])

                    points = np.column_stack((x, y, z))
                    colors = np.column_stack((r, g, b))

                    pcd = to_o3d_cloud(xyz=points, rgb=colors)

                    depth_image = rescale_greyscale(depth_image)

                    color_image_o3d = to_o3d_image(color_image)
                    depth_image_o3d = to_o3d_image(depth_image)

                def update():
                    self.rgb_widget.update_image(color_image_o3d)
                    self.depth_widget.update_image(depth_image_o3d)
                    # self.widget3d.scene.set_background(
                    #     [1, 1, 1, 1], color_image_o3d)
                    # pcd = make_point_cloud(100, (0, 0, 0), 1.0)
                    lit = rendering.Material()
                    lit.shader = "defaultLit"
                    if not self.widget3d.scene.has_geometry('pcd'):
                        self.widget3d.scene.add_geometry('pcd', pcd, lit)
                        # bounds = self.widget3d.scene.bounding_box
                        bounds = pcd.get_axis_aligned_bounding_box()
                        self.widget3d.setup_camera(
                            60.0, bounds, bounds.get_center())
                        # self.widget3d.scene.show_axes(True)
                    else:
                        self.widget3d.scene.remove_geometry('pcd')
                        self.widget3d.scene.add_geometry(
                            'pcd', pcd, lit)
                        # bounds = pcd.get_axis_aligned_bounding_box()
                        # self.widget3d.setup_camera(
                        #     60, bounds, bounds.get_center())

                if not self.is_done:
                    gui.Application.instance.post_to_main_thread(
                        self.window, update)


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    window = VideoWindow()

    app.run()


if __name__ == '__main__':
    main()
