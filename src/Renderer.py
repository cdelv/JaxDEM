# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions

import vtk
import jax
import jax.numpy as jnp
import numpy as np
import time
import copy

class Renderer:
    """
    Real-time visualization for JaxDEM simulations using VTK.
    
    This class provides an interactive 3D visualization of the simulation,
    with capabilities to pause, resume, and reset the simulation.
    
    Attributes
    ----------
    system : System
        The simulation system to render.
    window_size : tuple of int
        The size (width, height) of the render window in pixels.
    timer_interval : int
        The time interval in milliseconds between render updates.
    """
    
    # Corrected method order in the Renderer class

    def __init__(self, system, window_size=(800, 600), timer_interval=30):
        """
        Initialize the renderer.
        
        Parameters
        ----------
        system : System
            The JaxDEM simulation system to render.
        window_size : tuple of int, optional
            The size (width, height) of the render window in pixels.
        timer_interval : int, optional
            The time interval in milliseconds between render updates.
        """
        self.system = system
        self.window_size = window_size
        self.timer_interval = timer_interval
        
        # Flags for simulation control
        self.running = False
        self.paused = False
        
        # FPS tracking variables
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        
        # Store initial state for reset functionality
        self._store_initial_state()
        
        # Initialize VTK components
        self._init_vtk()
    
    def _store_initial_state(self):
        """Store the initial system state for later reset."""
        # Capture the initial state (pos, vel, accel) from memory
        initial_state = self.system.bodies.memory.getState()
        
        # Create deep copies to ensure we have independent copies
        self._initial_pos = jnp.array(initial_state[0])
        self._initial_vel = jnp.array(initial_state[1])
        self._initial_accel = jnp.array(initial_state[2])
        
        # Also store initial radius values
        self._initial_rad = jnp.array(self.system.bodies.memory._rad)
        
        # Store other necessary simulation properties
        self._initial_save_counter = self.system.saveCounter
    
    def _init_vtk(self):
        """Initialize VTK rendering components."""
        # Create renderer and render window
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.4)  # Dark blue background
        
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(*self.window_size)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("JaxDEM Simulation")
        
        # Interactor and interactor style
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Use trackball camera for easy 3D navigation
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Actor for domain visualization
        self._create_domain_actor()
        
        # Sphere glyph for particle visualization
        self._create_particle_visualization()
        
        # Setup HUD display for simulation info
        self._create_info_display()
        
        # Add title
        self._add_title("JaxDEM Simulation - Space: Pause/Resume, R: Reset")
        
        # Setup keyboard interaction (adding this line here instead of calling a separate method)
        self.interactor.AddObserver("KeyPressEvent", self._keyboard_callback)
        
        # Add help text for camera controls
        help_text = vtk.vtkTextActor()
        help_text.SetInput("Camera Controls: 1-XY plane, 2-YZ plane, 3-XZ plane")
        help_text.SetPosition(10, self.window_size[1] - 30)
        help_text.GetTextProperty().SetFontSize(14)
        help_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        self.renderer.AddActor2D(help_text)
        
        # Timer for animation
        self.timer_id = None
    
    def _create_domain_actor(self):
        """Create the domain visualization."""
        # Extract domain dimensions
        dim = int(self.system.dim)
        length = self.system.domain.length
        anchor = self.system.domain.anchor
        
        # Create a cube for the domain
        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(length[0]))
        cube.SetYLength(float(length[1]))
        
        center = [float(anchor[i] + length[i]/2) for i in range(dim)]
        
        if dim == 3:
            cube.SetZLength(float(length[2]))
            cube.SetCenter(center[0], center[1], center[2])
        else:  # dim == 2
            cube.SetZLength(0.1)  # Small thickness for 2D
            cube.SetCenter(center[0], center[1], 0.0)
        
        # Create mapper and actor for domain
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        
        self.domain_actor = vtk.vtkActor()
        self.domain_actor.SetMapper(mapper)
        self.domain_actor.GetProperty().SetOpacity(0.1)  # Make it transparent
        self.domain_actor.GetProperty().SetColor(0.8, 0.8, 1.0)  # Light blue
        self.domain_actor.GetProperty().SetRepresentationToWireframe()
        
        self.renderer.AddActor(self.domain_actor)
    
    def _create_particle_visualization(self):
        """Setup visualization for particles using sphere glyphs."""
        # Create points for sphere centers
        self.points = vtk.vtkPoints()
        
        # Create polydata to store points
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        
        # Create arrays for particle properties
        self.radius_array = vtk.vtkFloatArray()
        self.radius_array.SetName("Radius")
        
        self.velocity_array = vtk.vtkFloatArray()
        self.velocity_array.SetName("Velocity")
        self.velocity_array.SetNumberOfComponents(3)
        
        self.force_array = vtk.vtkFloatArray()
        self.force_array.SetName("Force")
        self.force_array.SetNumberOfComponents(3)
        
        # Add arrays to polydata
        self.polydata.GetPointData().AddArray(self.radius_array)
        self.polydata.GetPointData().AddArray(self.velocity_array)
        self.polydata.GetPointData().AddArray(self.force_array)
        self.polydata.GetPointData().SetActiveScalars("Radius")
        
        # Create sphere source for glyph
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetPhiResolution(20)
        sphere_source.SetThetaResolution(20)
        
        # Create the glyph
        self.glyph = vtk.vtkGlyph3D()
        self.glyph.SetSourceConnection(sphere_source.GetOutputPort())
        self.glyph.SetInputData(self.polydata)
        self.glyph.SetScaleModeToScaleByScalar()
        self.glyph.SetScaleFactor(1.0)
        self.glyph.SetColorModeToColorByScalar()
        
        # Create mapper and actor for particles
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph.GetOutputPort())
        
        # Setup a color map for the particles
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.67, 0.0)  # Blue to red
        lut.SetTableRange(0.0, 2.0)  # Range for radius values
        lut.Build()
        mapper.SetLookupTable(lut)
        
        self.particle_actor = vtk.vtkActor()
        self.particle_actor.SetMapper(mapper)
        
        # Add actor to renderer
        self.renderer.AddActor(self.particle_actor)
        
        # Create velocity vectors (optional)
        self._create_velocity_vectors()
    
    def _create_velocity_vectors(self):
        """Create arrows to visualize particle velocities."""
        # Create arrow source
        arrow = vtk.vtkArrowSource()
        
        # Create glyph for velocity visualization
        self.vel_glyph = vtk.vtkGlyph3D()
        self.vel_glyph.SetSourceConnection(arrow.GetOutputPort())
        self.vel_glyph.SetInputData(self.polydata)
        self.vel_glyph.SetVectorModeToUseVector()
        self.vel_glyph.SetScaleFactor(0.5)  # Adjust based on your velocity scale
        self.vel_glyph.SetColorModeToColorByVector()
        self.vel_glyph.OrientOn()
        self.vel_glyph.SetVectorModeToUseVector()
        self.vel_glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.vel_glyph.GetOutputPort())
        
        self.velocity_actor = vtk.vtkActor()
        self.velocity_actor.SetMapper(mapper)
        
        # Add actor to renderer
        self.renderer.AddActor(self.velocity_actor)
    
    def _create_info_display(self):
        """Create on-screen display for simulation information."""
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("Status: Ready\nTime: 0.0\nParticles: 0\nFPS: 0.0")
        self.text_actor.SetPosition(10, 10)
        self.text_actor.GetTextProperty().SetFontSize(14)
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        
        self.renderer.AddActor2D(self.text_actor)
    
    def _add_title(self, title):
        """Add a title to the visualization."""
        title_actor = vtk.vtkTextActor()
        title_actor.SetInput(title)
        title_actor.SetPosition(self.window_size[0] // 2 - 150, self.window_size[1] - 30)
        title_actor.GetTextProperty().SetFontSize(16)
        title_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        title_actor.GetTextProperty().SetJustificationToCentered()
        
        self.renderer.AddActor2D(title_actor)
    
    def _keyboard_callback(self, obj, event):
        """Handle keyboard interactions."""
        key = obj.GetKeySym().lower()
        
        if key == "space":
            # Toggle pause/resume
            self.paused = not self.paused
            status = "Paused" if self.paused else "Running"
            self._update_info_display(status=status)
            
        elif key == "r":
            # Reset simulation to initial state
            self._reset_simulation()
            
        elif key == "q":
            # Quit the simulation
            if self.timer_id is not None:
                self.interactor.DestroyTimer(self.timer_id)
            self.interactor.ExitCallback()
            
        elif key == "1":
            # Reset camera to XY plane (top view)
            self._reset_camera_to_xy_plane()
            
        elif key == "2":
            # Reset camera to YZ plane (side view)
            self._reset_camera_to_yz_plane()
            
        elif key == "3":
            # Reset camera to XZ plane (front view)
            self._reset_camera_to_xz_plane()
    
    def _update_info_display(self, status=None):
        """Update the information display."""
        if status is None:
            status = "Paused" if self.paused else "Running"
        
        # Calculate current simulation time
        current_time = self.system.saveCounter * float(self.system.saveTime)
        
        # Update the display text
        info_text = f"Status: {status}\n"
        info_text += f"Time: {current_time:.2f}\n"
        info_text += f"Particles: {self.system.bodies.nSpheres}\n"
        info_text += f"FPS: {self.fps:.1f}"
        
        self.text_actor.SetInput(info_text)
    
    def _update_visualization(self):
        """Update the visualization based on current system state."""
        # Clear previous points and data
        self.points.Reset()
        self.radius_array.Reset()
        self.velocity_array.Reset()
        self.force_array.Reset()
        
        # Get the current system state
        dim = int(self.system.dim)
        n_spheres = int(self.system.bodies.nSpheres)
        
        pos = self.system.bodies.memory._pos
        vel = self.system.bodies.memory._vel
        accel = self.system.bodies.memory._accel
        mass = self.system.bodies.memory._mass
        radius = self.system.bodies.memory._rad
        
        # Calculate forces from acceleration and mass
        force = jnp.multiply(accel, mass[:, jnp.newaxis])
        
        # Add points and data
        for i in range(n_spheres):
            # Extract as float for VTK compatibility
            p_pos = [float(pos[i, j]) for j in range(dim)]
            p_vel = [float(vel[i, j]) for j in range(dim)]
            p_force = [float(force[i, j]) for j in range(dim)]
            p_radius = float(radius[i])
            
            # Ensure 3D points for VTK
            if dim == 2:
                p_pos.append(0.0)
                p_vel.append(0.0)
                p_force.append(0.0)
            
            # Add point and data
            self.points.InsertNextPoint(p_pos)
            self.radius_array.InsertNextValue(p_radius)
            self.velocity_array.InsertNextTuple3(p_vel[0], p_vel[1], p_vel[2])
            self.force_array.InsertNextTuple3(p_force[0], p_force[1], p_force[2])
        
        # Mark data as modified to trigger update
        self.polydata.Modified()
        self.render_window.Render()
        
        # Increment frame counter for FPS calculation
        self.frame_count += 1
    
    def _timer_callback(self, obj, event):
        """Timer callback for animation."""
        # Calculate FPS
        self._update_fps()
        
        if not self.paused and self.running:
            # Perform one simulation step
            self._step_simulation()
            
            # Update visualization
            self._update_visualization()
            
            # Update info display
            self._update_info_display()
    
    def _step_simulation(self):
        """Perform a single simulation step."""
        # Access current state directly
        pos = self.system.bodies.memory._pos
        vel = self.system.bodies.memory._vel
        accel = self.system.bodies.memory._accel
        
        # Apply the stepping logic
        vel = vel + self.system.dt * accel
        pos = pos + self.system.dt * vel
        
        # Update the memory state
        self.system.bodies.memory.setState((pos, vel, accel))
        
        # Increment counter
        self.system.saveCounter += 1
    
    def _reset_simulation(self):
        """Reset the simulation to its initial state."""
        # Restore the initial state
        self.system.bodies.memory.setState((self._initial_pos, self._initial_vel, self._initial_accel))
        
        # Restore radius values
        self.system.bodies.memory._rad = self._initial_rad
        
        # Reset save counter
        self.system.saveCounter = self._initial_save_counter
        
        # Update visualization
        self._update_visualization()
        
        # Update status
        self._update_info_display(status="Reset")
        
        # Pause simulation
        self.paused = True
    
    def _update_fps(self):
        """Calculate and update the FPS counter."""
        current_time = time.time()
        time_elapsed = current_time - self.last_fps_time
        
        # Update FPS calculation every self.fps_update_interval seconds
        if time_elapsed >= self.fps_update_interval:
            self.fps = self.frame_count / time_elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def start(self):
        """Start the visualization and simulation."""
        # Initialize the visualization with current state
        self._update_visualization()
        
        # Set up the camera for optimal viewing
        self.renderer.ResetCamera()
        
        # Create a timer for animation
        self.interactor.Initialize()
        self.timer_id = self.interactor.CreateRepeatingTimer(self.timer_interval)
        self.interactor.AddObserver("TimerEvent", self._timer_callback)
        
        # Mark simulation as running
        self.running = True
        self.paused = False
        
        # Initialize FPS counter
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Start the interaction
        self.render_window.Render()
        self.interactor.Start()
    
    def stop(self):
        """Stop the visualization and simulation."""
        if self.timer_id is not None:
            self.interactor.DestroyTimer(self.timer_id)
        
        self.running = False
        self.paused = True
        
        # Clean up
        self.render_window.Finalize()
        self.interactor.TerminateApp()

    def _reset_camera_to_xy_plane(self):
        """Reset camera to view the XY plane (top view)."""
        camera = self.renderer.GetActiveCamera()
        
        # Get the center of the scene
        bounds = self.domain_actor.GetBounds()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        
        # Position the camera for XY plane (looking down the Z axis)
        camera.SetPosition(center_x, center_y, center_z + 20)
        camera.SetFocalPoint(center_x, center_y, center_z)
        camera.SetViewUp(0, 1, 0)
        
        self.renderer.ResetCamera()
        self.render_window.Render()
        
    def _reset_camera_to_yz_plane(self):
        """Reset camera to view the YZ plane (side view)."""
        camera = self.renderer.GetActiveCamera()
        
        # Get the center of the scene
        bounds = self.domain_actor.GetBounds()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        
        # Position the camera for YZ plane (looking down the X axis)
        camera.SetPosition(center_x + 20, center_y, center_z)
        camera.SetFocalPoint(center_x, center_y, center_z)
        camera.SetViewUp(0, 1, 0)
        
        self.renderer.ResetCamera()
        self.render_window.Render()
        
    def _reset_camera_to_xz_plane(self):
        """Reset camera to view the XZ plane (front view)."""
        camera = self.renderer.GetActiveCamera()
        
        # Get the center of the scene
        bounds = self.domain_actor.GetBounds()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        
        # Position the camera for XZ plane (looking down the Y axis)
        camera.SetPosition(center_x, center_y + 20, center_z)
        camera.SetFocalPoint(center_x, center_y, center_z)
        camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCamera()
        self.render_window.Render()