import vtk
import jax
import jax.numpy as jnp

import tkinter as tk
import time
import copy


class Renderer:
    """
    Real-time visualization for JaxDEM using VTK.
    
    This class provides an interactive 3D visualization of the simulation,
    with capabilities to pause, resume, and reset the simulation.
    
    Attributes
    ----------
    system : System
        The simulation system to render.
    windowSize : tuple of int
        The size (width, height) of the render window in pixels.
    timerInterval : int
        The time interval in milliseconds between render updates.

    TO DO: MAKE STEPING IN RENDERER MATCH WITH STEPING IN SYSTEM
    """
    
    def __init__(self, system, windowSize=2/3, timerInterval=15):
        """
        Initialize the renderer.
        
        Parameters
        ----------
        system : System
            The JaxDEM simulation system to render.
        windowSize : tuple of int, optional
            The size (width, height) of the render window in pixels.
        timerInterval : int, optional
            The time interval in milliseconds between render updates.
        """
        root = tk.Tk()
        root.withdraw()

        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()

        root.destroy()

        self.system = system
        self.windowSize = (int(screenWidth*windowSize), int(screenHeight*windowSize))
        self.timerInterval = timerInterval
        self.running = False
        self.paused = False
        self.fps = 0.0
        self.frameCount = 0
        self.lastFpsTime = time.time()
        self.fpsUpdateInterval = 0.5
        self._storeInitialState()
        self._initVtk()
    
    def _storeInitialState(self):
        """Store the initial system state for later reset."""
        initialState = self.system.bodies.memory.getState()
        
        self._initialPos = initialState[0]
        self._initialVel = initialState[1]
        self._initialAccel = initialState[2]
        self._initialRad = self.system.bodies.memory._rad
        self._initialSaveCounter = self.system.saveCounter

    def _initVtk(self):
        """Initialize VTK rendering components."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetSize(*self.windowSize)
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetWindowName("JaxDEM")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWindow)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        self._createDomainActor()
        self._createParticleVisualization()
        self._createInfoDisplay()
        
        titleActor = vtk.vtkTextActor()
        titleActor.SetInput("JaxDEM")
        titleActor.SetPosition(self.windowSize[0] // 2, self.windowSize[1] - 20)
        titleActor.GetTextProperty().SetFontSize(16)
        titleActor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        titleActor.GetTextProperty().SetJustificationToCentered()
        self.renderer.AddActor2D(titleActor)
        
        controlsText = vtk.vtkTextActor()
        controlsText.SetInput("Controls: Space-Pause/Resume, R-Reset, Q-Quit")
        controlsText.SetPosition(self.windowSize[0] - 10, 50)
        controlsText.GetTextProperty().SetFontSize(14)
        controlsText.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        controlsText.GetTextProperty().SetJustificationToRight()
        self.renderer.AddActor2D(controlsText)
        
        cameraText = vtk.vtkTextActor()
        cameraText.SetInput("Camera: 1-XY plane, 2-YZ plane, 3-XZ plane")
        cameraText.SetPosition(self.windowSize[0] - 10, 25)
        cameraText.GetTextProperty().SetFontSize(14)
        cameraText.GetTextProperty().SetColor(1.0, 1.0, 1.0) 
        cameraText.GetTextProperty().SetJustificationToRight()
        self.renderer.AddActor2D(cameraText)
        
        self.interactor.AddObserver("KeyPressEvent", self._keyboardCallback)
        self.timerId = None
    
    def _createDomainActor(self):
        """Create the domain visualization."""
        dim = int(self.system.dim)
        length = self.system.domain.length
        anchor = self.system.domain.anchor
        
        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(length[0]))
        cube.SetYLength(float(length[1]))
        center = [float(anchor[i] + length[i]/2) for i in range(dim)]

        if dim == 3:
            cube.SetZLength(float(length[2]))
            cube.SetCenter(center)
        else:
            cube.SetZLength(0.0)
            cube.SetCenter(center + [0.0])
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        
        self.domainActor = vtk.vtkActor()
        self.domainActor.SetMapper(mapper)
        self.domainActor.GetProperty().SetOpacity(0.1)
        self.domainActor.GetProperty().SetColor(0.8, 0.8, 1.0)
        self.domainActor.GetProperty().SetRepresentationToWireframe()
        
        self.renderer.AddActor(self.domainActor)
    
    def _createParticleVisualization(self):
        """Setup visualization for particles using sphere glyphs."""
        self.points = vtk.vtkPoints()
        
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        
        self.radiusArray = vtk.vtkFloatArray()
        self.radiusArray.SetName("Radius")
        
        self.polydata.GetPointData().AddArray(self.radiusArray)
        self.polydata.GetPointData().SetActiveScalars("Radius")
        
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetPhiResolution(32)
        sphereSource.SetThetaResolution(32)
        
        self.glyph = vtk.vtkGlyph3D()
        self.glyph.SetSourceConnection(sphereSource.GetOutputPort())
        self.glyph.SetInputData(self.polydata)
        self.glyph.SetScaleModeToScaleByScalar()
        self.glyph.SetScaleFactor(1.0)
        self.glyph.SetColorModeToColorByScalar()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph.GetOutputPort())
        
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.67, 0.0)
        lut.SetTableRange(0.0, 2.0)
        lut.Build()
        mapper.SetLookupTable(lut)
        
        self.particleActor = vtk.vtkActor()
        self.particleActor.SetMapper(mapper)
        
        self.renderer.AddActor(self.particleActor)
    
    def _createInfoDisplay(self):
        """Create on-screen display for simulation information."""
        self.textActor = vtk.vtkTextActor()
        self.textActor.SetInput("Status: Ready\nTime: 0.0\nParticles: 0\nFPS: 0.0")
        self.textActor.SetPosition(10, 10)
        self.textActor.GetTextProperty().SetFontSize(14)
        self.textActor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        
        self.renderer.AddActor2D(self.textActor)
    
    def _addTitle(self, title):
        """Add a title to the visualization."""
        titleActor = vtk.vtkTextActor()
        titleActor.SetInput(title)
        titleActor.SetPosition(self.windowSize[0] // 2 - 150, self.windowSize[1] - 30)
        titleActor.GetTextProperty().SetFontSize(16)
        titleActor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        titleActor.GetTextProperty().SetJustificationToCentered()
        
        self.renderer.AddActor2D(titleActor)
    
    def _keyboardCallback(self, obj, event):
        """Handle keyboard interactions."""
        key = obj.GetKeySym().lower()
        
        if key == "space":
            self.paused = not self.paused
            status = "Paused" if self.paused else "Running"
            self._updateInfoDisplay(status=status)
            
        elif key == "r":
            self._resetSimulation()
            
        elif key == "q":
            if self.timerId is not None:
                self.interactor.DestroyTimer(self.timerId)
            self.interactor.ExitCallback()
            
        elif key == "1":
            self._resetCameraToXyPlane()
            
        elif key == "2":
            self._resetCameraToYzPlane()
            
        elif key == "3":
            self._resetCameraToXzPlane()
    
    def _updateInfoDisplay(self, status=None):
        """Update the information display."""
        if status is None:
            status = "Paused" if self.paused else "Running"
        
        currentTime = self.system.saveCounter * float(self.system.saveTime)
        
        infoText = f"Status: {status}\n"
        infoText += f"Time: {currentTime:.2f}\n"
        infoText += f"Particles: {self.system.bodies.nSpheres}\n"
        infoText += f"FPS: {self.fps:.1f}"
        
        self.textActor.SetInput(infoText)
    
    def _updateVisualization(self):
        """Update the visualization based on current system state."""
        self.points.Reset()
        self.radiusArray.Reset()
        
        dim = int(self.system.dim)
        nSpheres = int(self.system.bodies.nSpheres)
        
        pos = self.system.bodies.memory._pos
        radius = self.system.bodies.memory._rad
        
        # Prepare arrays for batch operations
        positions = []
        radii = []
        
        # Extract data for all particles
        for i in range(nSpheres):
            pPos = [float(pos[i, j]) for j in range(dim)]
            pRadius = float(radius[i])
            
            # Ensure 3D points for VTK
            if dim == 2:
                pPos.append(0.0)
            
            positions.append(pPos)
            radii.append(pRadius)
        
        # Batch insert all points
        for pPos in positions:
            self.points.InsertNextPoint(pPos)
        
        # Batch insert all data arrays
        for pRadius in radii:
            self.radiusArray.InsertNextValue(pRadius)
        
        self.polydata.Modified()
        self.renderWindow.Render()
        
        self.frameCount += 1
    
    def _timerCallback(self, obj, event):
        """Timer callback for animation."""
        self._updateFps()
        
        if not self.paused and self.running:
            self._stepSimulation()
            self._updateVisualization()
            self._updateInfoDisplay()
    
    def _stepSimulation(self):
        """
        Perform a single simulation step.
        TO DO: CALL STEP FROM SYSTEM
        """
        pos = self.system.bodies.memory._pos
        vel = self.system.bodies.memory._vel
        accel = self.system.bodies.memory._accel
        
        vel = vel + self.system.dt * accel
        pos = pos + self.system.dt * vel
        
        self.system.bodies.memory.setState((pos, vel, accel))
        
        self.system.saveCounter += 1
    
    def _resetSimulation(self):
        """
        Reset the simulation to its initial state.
        TO DO: FIND A BETTER WAY OF DOING THIS
        """
        self.system.bodies.memory.setState((self._initialPos, self._initialVel, self._initialAccel))
        self.system.bodies.memory._rad = self._initialRad
        self.system.saveCounter = self._initialSaveCounter
        
        self._updateVisualization()
        self._updateInfoDisplay(status="Reset")
        self.paused = True
    
    def _updateFps(self):
        """Calculate and update the FPS counter."""
        currentTime = time.time()
        timeElapsed = currentTime - self.lastFpsTime
        
        if timeElapsed >= self.fpsUpdateInterval:
            self.fps = self.frameCount / timeElapsed
            self.frameCount = 0
            self.lastFpsTime = currentTime
    
    def start(self):
        """Start the visualization and simulation."""
        self._updateVisualization()
        self.renderer.ResetCamera()
        self.interactor.Initialize()
        self.timerId = self.interactor.CreateRepeatingTimer(self.timerInterval)
        self.interactor.AddObserver("TimerEvent", self._timerCallback)
        self.running = True
        self.paused = False
        self.frameCount = 0
        self.lastFpsTime = time.time()
        self.renderWindow.Render()
        self.interactor.Start()
    
    def stop(self):
        """Stop the visualization and simulation."""
        if self.timerId is not None:
            self.interactor.DestroyTimer(self.timerId)
        
        self.running = False
        self.paused = True
        
        self.renderWindow.Finalize()
        self.interactor.TerminateApp()

    def _resetCameraToXyPlane(self):
        """Reset camera to view the XY plane (top view)."""
        camera = self.renderer.GetActiveCamera()
        
        bounds = self.domainActor.GetBounds()
        centerX = (bounds[0] + bounds[1]) / 2
        centerY = (bounds[2] + bounds[3]) / 2
        centerZ = (bounds[4] + bounds[5]) / 2
        
        camera.SetPosition(centerX, centerY, centerZ + 20)
        camera.SetFocalPoint(centerX, centerY, centerZ)
        camera.SetViewUp(0, 1, 0)
        
        self.renderer.ResetCamera()
        self.renderWindow.Render()
        
    def _resetCameraToYzPlane(self):
        """Reset camera to view the YZ plane (side view)."""
        camera = self.renderer.GetActiveCamera()
        
        bounds = self.domainActor.GetBounds()
        centerX = (bounds[0] + bounds[1]) / 2
        centerY = (bounds[2] + bounds[3]) / 2
        centerZ = (bounds[4] + bounds[5]) / 2
        
        camera.SetPosition(centerX + 20, centerY, centerZ)
        camera.SetFocalPoint(centerX, centerY, centerZ)
        camera.SetViewUp(0, 1, 0)
        
        self.renderer.ResetCamera()
        self.renderWindow.Render()
        
    def _resetCameraToXzPlane(self):
        """Reset camera to view the XZ plane (front view)."""
        camera = self.renderer.GetActiveCamera()
        
        bounds = self.domainActor.GetBounds()
        centerX = (bounds[0] + bounds[1]) / 2
        centerY = (bounds[2] + bounds[3]) / 2
        centerZ = (bounds[4] + bounds[5]) / 2
        
        camera.SetPosition(centerX, centerY + 20, centerZ)
        camera.SetFocalPoint(centerX, centerY, centerZ)
        camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCamera()
        self.renderWindow.Render()