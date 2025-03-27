""""
PORTED OVER INVESTIGATION FROM CAPSTONE PROJECT - Spherical Coordinate stuff does not apply
File was written to help visualize and verify the math being used for camera projection
    * The code includes functionality for:
        * Creating a camera object with a given focal length, resolution, pixel size, radial distance from positioning centre
        * Projecting a point from the camera frame to the image frame and vice versa
        * Transforming from camera frame to world frame and vice versa
    * Math in this code is validated, and supports the method for estimating projection commands
"""
import numpy as np

def ft_to_mm(ft):
    return ft*304.8

room_width = ft_to_mm(15)
room_height = ft_to_mm(10)

def generate_box(center, width, height, depth):
    return [center[0] - width/2, center[1] - height/2, center[2], 1],\
        [center[0] + width/2, center[1] - height/2, center[2], 1],\
        [center[0] + width/2, center[1] + height/2, center[2], 1], \
        [center[0] - width/2, center[1] + height/2, center[2], 1], \
        [center[0] - width/2, center[1] - height/2, center[2] + depth, 1], \
        [center[0] + width/2, center[1] + height/2, center[2] + depth, 1], \
        [center[0] - width/2, center[1] + height/2, center[2] + depth, 1], \
        [center[0] + width/2, center[1] - height/2, center[2] + depth, 1], \
    

class Camera:
    def __init__(self, focal_length, resX, resY, radial_distance, pixel_size):
        self.focal_length = focal_length #Focal Length in mm
        self.resX = resX # Resolution in Pixels
        self.resY = resY # Resolution in Pixels
        self.pixel_size= pixel_size #Pixel size in mm
        self.radial_distance = radial_distance

        self.sensor_width = resX * pixel_size
        self.sensor_height = resY * pixel_size
        self.T_cam_photo = self.calculate_T_cam_photo()
        self.FOV = self.calculate_fov()
        self.calculate_T_cam_world(0, 0)

    #Function to project out a point into the 3D reference frame of the camera
    def calculate_T_cam_photo(self):
        return np.array([
            [self.focal_length/self.pixel_size, 0, self.resX/2],
            [0, self.focal_length/self.pixel_size, self.resY/2],
            [0, 0, 1]
        ])
    
    #Change from 3D point in camera reference frame to 2D point in image reference frame
    def cam_to_image(self, point):
        return np.dot(self.T_cam_photo, point[:3]/point[2])

    def image_to_cam(self, point, distance):
        return np.dot(np.linalg.inv(self.T_cam_photo), point)*distance
    
    def image_to_sensor_plane(self, point):
        return self.image_to_cam(point, self.focal_length)

    def calculate_fov(self):
        return [np.rad2deg(2*np.arctan(x/(2*self.focal_length))) for x in [
            self.sensor_width,
            self.sensor_height,
            np.sqrt(self.sensor_width**2 + self.sensor_height**2)
            ]]
    
    #Calculate matrix to transform camera coordinates to world coordinates (homogenous)
    def calculate_T_cam_world(self, theta, phi):
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)

        T_cam_world = np.linalg.inv(np.array([
            [np.sin(theta_rad), -np.cos(theta_rad), 0, 0],
            [-np.sin(phi_rad) * np.cos(theta_rad), -np.sin(phi_rad) * np.sin(theta_rad), -np.cos(phi_rad), 0],
            [np.cos(phi_rad) * np.cos(theta_rad), np.cos(phi_rad) * np.sin(theta_rad), -np.sin(phi_rad), -self.radial_distance],
            [0, 0, 0, 1]
        ]))

        self.T_cam_world = T_cam_world
    
    def estimate_theta_phi_abs(self, point, assumed_dist=ft_to_mm(14.25)):
        #Project the point out the distance
        cam_point = np.append(self.image_to_cam(point, assumed_dist - self.radial_distance), 1)
        #Project the point out to the world frame
        world_point = np.dot(self.T_cam_world, cam_point)
        #Calculate the angles
        phi = np.arcsin(-world_point[2]/np.linalg.norm(world_point[:3]))
        theta = np.arctan2(world_point[1], world_point[0])

        return theta, phi
    
if __name__ == "__main__":
    #Create a camera object
    camera = Camera(4.74, 4608, 2592, 50, 1.4e-3)
    print(camera.FOV)

    ##VISUALIZATION CODE##
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.widgets import Slider

    #Create a plot, 3d in the left pane, 2d in the right pane
    fig = plt.figure()
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # Add lines to pass through the origin for each axis
    ax3d.plot([-1000, 1000], [0, 0], [0, 0], 'k--', alpha=0.5)  # X-axis
    ax3d.plot([0, 0], [-1000, 1000], [0, 0], 'k--', alpha=0.5)  # Y-axis
    ax3d.plot([0, 0], [0, 0], [-1000, 1000], 'k--', alpha=0.5)  # Z-axis

    # Setting axis limits
    ax3d.set_xlim([0, room_width])
    ax3d.set_ylim([0, room_width])
    ax3d.set_zlim([-room_height, 0])
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')


    ax2d.set_xlim([0, camera.resX])
    ax2d.set_ylim([-camera.resY, 0])
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    
    #Set titles
    ax3d.set_title('3D Representation of Camera FOV')
    ax2d.set_title('2D Representation of Camera FOV')

    # Define the four corners of the image frame in the camera image frame
    image_corners = np.array([
        [0, 0, 1],  # Bottom-left
        [camera.resX, 0, 1],   # Bottom-right
        [camera.resX, camera.resY, 1],    # Top-right
        [0, camera.resY, 1]    # Top-left
    ])

    #project out at a distance of 100mm from the camera in the camera z direction
    cone_depth = 500
    cam_corners = [np.append(camera.image_to_cam(corner, cone_depth), 1) for corner in image_corners]
    
    #Transform the corners to the world frame
    transformed_corners = [np.dot(camera.T_cam_world, corner) for corner in cam_corners]
    transformed_camera_origin = np.dot(camera.T_cam_world, np.array([0, 0, 0, 1]))
    
    #Point in the world to represent an object
    w = ft_to_mm(5)

    world_objs = []
    box_locations = [
        [room_width - w/2, room_width-w/2, -room_height],
        [w/2, room_width-w/2, -room_height],
        [room_width - w/2, w/2, -room_height],
        [w/2, w/2, -room_height]
    ]
    for location in box_locations:
        for box in generate_box(location, w, w, w):
            world_objs.append(box)

    #Plot the world obj
    for world_obj in world_objs:
        ax3d.scatter(world_obj[0], world_obj[1], world_obj[2], c='r', s=50)

    #plot lines connecting the frame
    image_frame, = ax3d.plot(
        [corner[0] for corner in transformed_corners] + [transformed_corners[0][0]], 
        [corner[1] for corner in transformed_corners] + [transformed_corners[0][1]],
        [corner[2] for corner in transformed_corners] + [transformed_corners[0][2]],
        'b-'
    )
    
    # Plot the vector from the origin to the world point
    vector, = ax3d.plot([0, transformed_camera_origin[0]], [0, transformed_camera_origin[1]], [0, transformed_camera_origin[2]], 'g-')

    #plot lines connecting camera origin to each of the transformed corners
    image_cone, = ax3d.plot(
        np.array([[transformed_camera_origin[0], corner[0]] for corner in transformed_corners]).flatten(),
        np.array([[transformed_camera_origin[1], corner[1]] for corner in transformed_corners]).flatten(),
        np.array([[transformed_camera_origin[2], corner[2]] for corner in transformed_corners]).flatten(),
        'b-'
    )

    #Camera view of world object
    cam_objs = np.array([np.dot(np.linalg.inv(camera.T_cam_world), world_obj) for world_obj in world_objs])
    cam_objs = np.array([camera.cam_to_image(cam_obj[:3]) for cam_obj in cam_objs])
    photo_rep = ax2d.scatter([cam_obj[0] for cam_obj in cam_objs],
                            [-cam_obj[1] for cam_obj in cam_objs],
                            c='r',
                            s=50)
    
    #Back Calculate estimated theta and phi
    spherical_coords = np.array([camera.estimate_theta_phi_abs(cam_obj) for cam_obj in cam_objs if (cam_obj[0] < camera.resX and cam_obj[0] > 0 and cam_obj[1] > 0 and cam_obj[1] < camera.resY)])

    #Project rays out using spherical coordinates
    rays = np.array([[
        np.cos(c[0])*np.cos(c[1]),
        np.sin(c[0])*np.cos(c[1]),
        -np.sin(c[1])] for c in spherical_coords])*ft_to_mm(25)
        
    ray_plotter, = ax3d.plot(
        np.array([[0, ray[0]] for ray in rays]).flatten(),
        np.array([[0, ray[1]] for ray in rays]).flatten(),
        np.array([[0, ray[2]] for ray in rays]).flatten(),
        'black'
    ) 

    # Create sliders for theta and phi
    ax_theta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_phi = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    theta = 0
    phi = 0
    slider_theta = Slider(ax_theta, 'Theta', 0, 360, valinit=theta)
    slider_phi = Slider(ax_phi, 'Phi', -20, 90, valinit=phi)

    def update(val):
        theta = slider_theta.val
        phi = slider_phi.val
        camera.calculate_T_cam_world(theta, phi)

        #Update Camera Frame in world coordinates
        transformed_corners = [np.dot(camera.T_cam_world, corner) for corner in cam_corners]
        transformed_camera_origin = np.dot(camera.T_cam_world, np.array([0, 0, 0, 1]))

        #Update the 2D representation of the object
        cam_obj = np.dot(np.linalg.inv(camera.T_cam_world), world_obj)
        cam_obj = camera.cam_to_image(cam_obj[:3])
        
        #Update the 3D plot
        #Vector
        vector.set_data([0, transformed_camera_origin[0]], [0, transformed_camera_origin[1]])
        vector.set_3d_properties([0, transformed_camera_origin[2]])

        #Image Frame
        image_frame.set_data(
            [corner[0] for corner in transformed_corners] + [transformed_corners[0][0]], 
            [corner[1] for corner in transformed_corners] + [transformed_corners[0][1]]
        )
        image_frame.set_3d_properties(
            [corner[2] for corner in transformed_corners] + [transformed_corners[0][2]]
        )

        #image cone
        image_cone.set_data(
            np.array([[transformed_camera_origin[0], corner[0]] for corner in transformed_corners]).flatten(),
            np.array([[transformed_camera_origin[1], corner[1]] for corner in transformed_corners]).flatten()
        )
        image_cone.set_3d_properties(
            np.array([[transformed_camera_origin[2], corner[2]] for corner in transformed_corners]).flatten()
        )
        
        #Update the 2D plot
        cam_objs = np.array([np.dot(np.linalg.inv(camera.T_cam_world), world_obj) for world_obj in world_objs])
        cam_objs = np.array([camera.cam_to_image(cam_obj[:3]) for cam_obj in cam_objs])
        photo_rep.set_offsets(np.c_[cam_objs[:, 0], -cam_objs[:, 1]])

            #Back Calculate estimated theta and phi
        spherical_coords = np.array([camera.estimate_theta_phi_abs(cam_obj) for cam_obj in cam_objs if (cam_obj[0] < camera.resX and cam_obj[0] > 0 and cam_obj[1] > 0 and cam_obj[1] < camera.resY)])

        #Project rays out using spherical coordinates
        rays = np.array([[
            np.cos(c[0])*np.cos(c[1]),
            np.sin(c[0])*np.cos(c[1]),
            -np.sin(c[1])] for c in spherical_coords])*ft_to_mm(25)
            
        ray_plotter.set_data(
            np.array([[0, ray[0]] for ray in rays]).flatten(),
            np.array([[0, ray[1]] for ray in rays]).flatten()
        )
        ray_plotter.set_3d_properties(
            np.array([[0, ray[2]] for ray in rays]).flatten()
        )

        fig.canvas.draw_idle()


    # Connect sliders to update function
    slider_theta.on_changed(update)
    slider_phi.on_changed(update)
    plt.show()
    