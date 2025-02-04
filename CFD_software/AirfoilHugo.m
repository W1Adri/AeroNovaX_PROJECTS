%{
-------------------------------------------------------------------------------------------
 Coder:       <Hugo García & Adrià Sancho>
 Date:        <20/01/2025>
 Course:      Fluid Mechanics (Potential Flow / Laplace solver)
 Description: This code simulates incompressible, inviscid, steady flow around either a 
              blunt circle or a streamlined airfoil. It solves Laplace’s equation for the
              stream function ψ by a simple relaxation (Jacobi), then derives velocity
              (u, v) and pressure (p). Finally, it integrates the pressure over the body’s
              boundary to compute the net force. 
-------------------------------------------------------------------------------------------
%}

clear; close all; clc;

%% --------------------------- USER SETTINGS ----------------------------
iterations             = 500;    % Number of iterations for the streamfunction solver
num_points            = 1000;    % Number of points to approximate the shape of the body
figure_to_be_analyzed = "streamlined"; 
% "streamlined" or "blunt" bodies

% Domain size (smaller domain => more forced flow around the shape => higher forces)
Lx = 30.0;  % Domain length in x-direction
Ly = 30.0;  % Domain length in y-direction

% Grid resolution (Nx x Ny)
Nx = 300;   % Number of points in the x-direction
Ny = 300;   % Number of points in the y-direction

% Grid spacing
dx = Lx / (Nx - 1);
dy = Ly / (Ny - 1);

% Fluid / Flow constants
density = 1.225;      % Density [kg/m^3]
P0      = 101325;     % Free-stream (atmospheric) pressure [Pa]
V0      = 30.0;       % Free-stream velocity [m/s]

%% --------------------------- CREATE GRID ----------------------------
% Coordinate arrays:
x_array = linspace(0, Lx, Nx);  % "Nx" x-coordinates from 0 to Lx
y_array = linspace(0, Ly, Ny);  % "Ny" y-coordinates from 0 to Ly

% We can create a mesh (xx, yy) for plotting or for referencing coordinates:
[XX, YY] = meshgrid(x_array, y_array);
% XX(j,i) will give the x-coordinate at row j, column i
% YY(j,i) will give the y-coordinate at row j, column i

%% --------------------------- ALLOCATE ARRAYS ----------------------------
% We store fields as Ny x Nx to be consistent with (j,i) indexing:
psi      = zeros(Ny, Nx);  % Streamfunction
u        = zeros(Ny, Nx);  % Velocity component in x
v        = zeros(Ny, Nx);  % Velocity component in y
p        = zeros(Ny, Nx);  % Pressure
velocity = zeros(Ny, Nx);  % Magnitude of velocity

%% --------------------------- DEFINE THE BODY ----------------------------
% We will create a function to produce the body contour points in (x, y).

bodycontour = crear_contorno(figure_to_be_analyzed, num_points, Lx, Ly);
% contour will be a (num_points, 2) array with the (x,y) coordinates of the shape
% in CCW (counterclockwise) order.

% We want to find which (j,i) grid points are INSIDE the body. 
% We'll use MATLAB's inpolygon for that.

body = zeros(Ny, Nx);      % Will mark 1 for inside the body
fluid = [];                % Will list fluid points (j,i)
boundary = [];             % Will list boundary points (j,i)

% Because we have Nx columns (x-direction) and Ny rows (y-direction), 
% we loop over i=1..Nx, j=1..Ny to test each cell center.

% We skip the domain edges):
for i = 2:(Nx-1)
    for j = 2:(Ny-1)
        % cell-center approx: (cx, cy) 
        % In row j, column i => x = x_array(i), y = y_array(j).
        cx = x_array(i);
        cy = y_array(j);

        % inpolygon(cx, cy, contour_x, contour_y) => true if (cx, cy) is inside the polygon
        inside = inpolygon(cx, cy, bodycontour(:,1), bodycontour(:,2));
        
        if inside
            % If inside the body, mark as body
            body(j, i) = 1;  
            boundary = [boundary; j, i];
            % The original Python code appended everything inside to boundary,
            % but that is actually the entire interior, not just the perimeter!
            % We'll keep it identical to the original code. 
        else
            % Otherwise, it's a fluid cell
            fluid = [fluid; j, i];
        end
    end
end

%% --------- IDENTIFY FLUID CELLS THAT ARE ADJACENT TO THE BODY (CONTACT SURFACE) -----------
contact_surface = [];
neighbors_4 = [ 1, 0; -1, 0; 0, 1; 0, -1 ]; % 4-direction neighbors

% For each fluid cell, check if any neighbor is in the body => contact surface
for idx = 1:size(fluid, 1)
    j = fluid(idx, 1);
    i = fluid(idx, 2);
    
    % Check the 4 neighbors
    is_contact = false;
    for nn = 1:4
        jn = j + neighbors_4(nn,1);
        in_ = i + neighbors_4(nn,2);
        
        % Make sure neighbor is within array bounds
        if (jn >= 1 && jn <= Ny && in_ >= 1 && in_ <= Nx)
            if body(jn, in_) == 1
                is_contact = true;
                break;
            end
        end
    end
    
    if is_contact
        contact_surface = [contact_surface; j, i];
    end
end

%% --------------------------- BOUNDARY CONDITIONS FOR ψ ----------------------------
% We want uniform flow from left to right: 
%   ψ(x,0) = 0, ψ(x, Ly) = V0*Ly,
%   ψ(0,y) = V0*y, ψ(Lx,y) = V0*y

% Bottom boundary: j=1
psi(1, :) = 0.0;

% Top boundary: j=Ny
psi(Ny, :) = V0 * Ly;

% Left + Right boundaries: i=1, i=Nx
for j = 1:Ny
    psi(j, 1)   = V0 * dy * (j-1);    % j-1 because j=1 => y=0
    psi(j, Nx)  = V0 * dy * (j-1);
end

% Initialize the interior fluid guess:
for idx = 1:size(fluid, 1)
    j = fluid(idx, 1);
    i = fluid(idx, 2);
    psi(j, i) = V0 * dy * (j-1);
end

% The contact surface on the body is "constant-psi" because it's a streamline
psi_vals = zeros(size(contact_surface,1),1);
for idx = 1:size(contact_surface,1)
    j = contact_surface(idx,1);
    i = contact_surface(idx,2);
    psi_vals(idx) = psi(j,i);
end
surface_psi = mean(psi_vals);

% Force body boundary to have that same psi
for idx = 1:size(boundary,1)
    j = boundary(idx,1);
    i = boundary(idx,2);
    psi(j,i) = surface_psi;
end

%% --------------------------- RELAXATION TO SOLVE LAPLACE(ψ)=0 ----------------------
% Jacobi iteration for ∂²ψ/∂x² + ∂²ψ/∂y² = 0

for it = 1:iterations
    % Update interior fluid cells
    for idx = 1:size(fluid,1)
        j = fluid(idx,1);
        i = fluid(idx,2);
        
        % Jacobi formula:
        % ψ_new = 0.5 * ((ψ_e + ψ_w)/dx^2 + (ψ_n + ψ_s)/dy^2) * (dx^2*dy^2/(dx^2+dy^2))
        
        psiE = psi(j, i+1);
        psiW = psi(j, i-1);
        psiN = psi(j+1, i);
        psiS = psi(j-1, i);
        
        psi(j, i) = 0.5 * ...
                    ( (psiE + psiW)/dx^2 + (psiN + psiS)/dy^2 ) * ...
                    ( dx^2 * dy^2 / (dx^2 + dy^2) );
    end
    
    % After updating ψ, compute velocity and pressure in the fluid region
    for idx = 1:size(fluid,1)
        j = fluid(idx,1);
        i = fluid(idx,2);
        
        % Central differences in y for u = ∂ψ/∂y
        if (j>1 && j<Ny)
            u(j, i) = (psi(j+1, i) - psi(j-1, i)) / (2 * dy);
        end
        
        % Central differences in x for v = -∂ψ/∂x
        if (i>1 && i<Nx)
            v(j, i) = -(psi(j, i+1) - psi(j, i-1)) / (2 * dx);
        end
        
        % Velocity magnitude
        velocity(j, i) = sqrt(u(j,i)^2 + v(j,i)^2);
        
        % Bernoulli's principle (assuming constant across the domain):
        % p + 0.5*rho*(u^2 + v^2) = constant 
        % Use P0 as reference at free stream => p = P0 - 0.5*rho*(u^2+v^2) + 0.5*rho*V0^2 - ...
        % For simplicity, just do:
        p(j, i) = P0 - 0.5 * density * ( u(j,i)^2 + v(j,i)^2 );
    end
    
    % Keep the body boundary on the same single ψ-value
    psi_vals = zeros(size(contact_surface,1),1);
    for cidx = 1:size(contact_surface,1)
        jtmp = contact_surface(cidx,1);
        itmp = contact_surface(cidx,2);
        psi_vals(cidx) = psi(jtmp,itmp);
    end
    surface_psi = mean(psi_vals);
    
    for bidx = 1:size(boundary,1)
        jtmp = boundary(bidx,1);
        itmp = boundary(bidx,2);
        psi(jtmp,itmp) = surface_psi;
    end
    
    % If desired, print iteration progress:
    % fprintf('Iteration %d/%d done.\n', it, iterations);
end

%% --------------------------- COMPUTE NET FORCE FROM PRESSURE ------------------------
% We now want to integrate p * n over the body boundary, where n is the outward normal.

% 1) We need to reconstruct the boundary in a continuous loop for integration. 
%    The original Python code uses "contour" for that. 
%    Then it finds the nearest grid indices for each segment.

boundary_segments = [];  % will store pairs of ((j1,i1), (j2,i2))

for k = 1:size(bodycontour,1)
    % segment from contour(k,:) to contour(k+1,:), wrapping around
    % in MATLAB, we handle the wrap by using mod or an if-statement:
    if k < size(bodycontour,1)
        p1 = bodycontour(k, :);
        p2 = bodycontour(k+1, :);
    else
        p1 = bodycontour(k, :);
        p2 = bodycontour(1, :);  % wrap to first point
    end
    
    % Convert these real (x,y) to nearest grid indices
    [j1, i1] = nearest_grid_index( p1(1), p1(2), dx, dy );
    [j2, i2] = nearest_grid_index( p2(1), p2(2), dx, dy );
    
    % Store them
    boundary_segments = [boundary_segments; j1, i1, j2, i2];
end

% Now sum up panel forces
net_force = [0.0, 0.0];

for segID = 1:size(boundary_segments,1)
    j1 = boundary_segments(segID,1);
    i1 = boundary_segments(segID,2);
    j2 = boundary_segments(segID,3);
    i2 = boundary_segments(segID,4);
    
    % length of the segment in the grid
    ds = panel_length([j1, i1],[j2, i2], dx, dy);
    % outward normal
    n = outward_normal([j1, i1],[j2, i2]);
    
    % local pressure => use midpoint's pressure
    pmid = p(j1, i1);
    
    % dF from pressure = -p * n * ds  (pressure acts inward, normal is outward)
    dF = -pmid * ds * n;
    net_force = net_force + dF;
end

Fx   = net_force(1);
Fy   = net_force(2);
F_mag = norm(net_force);
F_ang = atan2d(Fy, Fx);

fprintf("------------------------------------------------------\n");
fprintf("  Resultant Force:  %.2f  N\n", F_mag);
fprintf("  Components:       Fx = %.2f  N,   Fy = %.2f  N\n", Fx, Fy);
fprintf("  Direction Angle:  %.2f degrees from +x axis\n", F_ang);
fprintf("------------------------------------------------------\n");

%% --------------------------- PLOTTING ------------------------------
% Create a mask for the fluid region (to avoid plotting inside the body).
% We can do something similar to the Python code: if it's fluid => true, else => false.

mask = false(Ny, Nx);
for idx = 1:size(fluid,1)
    j = fluid(idx,1);
    i = fluid(idx,2);
    mask(j,i) = true;
end

% 1) Streamfunction
figure(1);
% We want to plot psi only where fluid is true; otherwise NaN
psi_plot = psi;
psi_plot(~mask) = NaN;  
contourf(XX, YY, psi_plot, 50, 'LineColor', 'none');
colorbar; hold on;
contour(XX, YY, psi_plot, 50, 'k', 'LineWidth', 0.5);
title("Streamfunction \psi");
xlabel("x [m]"); ylabel("y [m]");
axis equal; axis tight;
%sgtitle(sprintf("Flow around a %s body (Potential Flow)", figure_to_be_analyzed));

% 2) Pressure Field + net force vector
figure(2);
p_plot = p;
p_plot(~mask) = NaN;
contourf(XX, YY, p_plot, 50, 'LineColor', 'none');
colorbar; hold on;
title(sprintf("Pressure Field [Pa]. Free-stream = %g Pa", P0));
xlabel("x [m]"); ylabel("y [m]");
axis equal; axis tight;
%sgtitle(sprintf("Incompressible, steady, inviscid flow around a %s body.", figure_to_be_analyzed));

% Choose a convenient point for drawing the force arrow, e.g. near the body center
body_center_x = 0.5 * ( min(bodycontour(:,1)) + max(bodycontour(:,1)) );
body_center_y = 0.5 * ( min(bodycontour(:,2)) + max(bodycontour(:,2)) );

% Normalized direction of net force
if (F_mag < 1e-12)
    dir_arrow = [0, 0];
else
    dir_arrow = net_force / F_mag;
end

% We'll scale the arrow so it's visible. Let's choose an arbitrary scale factor:
scale_arrow = 0.2; 
quiver(body_center_x, body_center_y, ...
       dir_arrow(1), dir_arrow(2), ...
       scale_arrow, 'Color','red','LineWidth',1.5, 'MaxHeadSize',2);
text(body_center_x, body_center_y, ...
     sprintf("\\leftarrow |F|=%.2f N, \\theta=%.2f^\\circ", F_mag, F_ang), ...
     'Color','red','FontWeight','bold');

% 3) Velocity Field
figure(3);
v_plot = velocity;
v_plot(~mask) = NaN;
contourf(XX, YY, v_plot, 50, 'LineColor', 'none');
colorbar; hold on;
title(sprintf("Velocity Magnitude [m/s], Free-stream = %g m/s", V0));
xlabel("x [m]"); ylabel("y [m]");
axis equal; axis tight;
%sgtitle(sprintf("Incompressible, inviscid, steady flow around a %s body.", figure_to_be_analyzed));

%% ========================== SUB-FUNCTIONS ===========================
function contour_final = crear_contorno(name, num_points, Lx, Ly)
%{
  Creates a set of (x,y) points that trace the body in a SINGLE,
  CONSISTENT (counterclockwise) direction around its perimeter.
  For a circle, it's straightforward. For an airfoil, ensure 
  we go from TE->LE along the upper surface, then LE->TE along
  the lower surface in CCW order.
%}
    if name == "blunt"
        % A circle in the domain's center
        center = [Lx/2, Ly/2];
        radius = 1.5;
        theta  = linspace(0, 2*pi, num_points);
        
        % We'll go around the circle CCW
        xvals = center(1) + radius * cos(theta);
        yvals = center(2) + radius * sin(theta);
        contour_final = [xvals(:), yvals(:)];
        
    elseif name == "streamlined"
        % A simple "NACA-like" shape
        chord_length = 5.0;
        thickness    = 0.75;
        
        % Parametric x from 0 to 1
        x_local = linspace(0, 1, num_points/2);
        
        % Standard approximate NACA formula for thickness distribution
        yt = 5 * thickness * ( ...
             0.2969*sqrt(x_local) - 0.1260*x_local - 0.3516*x_local.^2 + ...
             0.2843*x_local.^3    - 0.1036*x_local.^4 );
        
        % Coordinates for top (upper surface),
        %   going from TE (x=1) to LE (x=0) in CCW order
        x_upper = x_local(end:-1:1) * chord_length;  % reversed in x
        y_upper = +yt(end:-1:1);
        
        % Coordinates for bottom (lower surface),
        %   going from LE (x=0) to TE (x=1)
        x_lower = x_local * chord_length;
        y_lower = -yt;
        
        % Shift so chord is centered in domain (x)
        shift_x = (Lx/2 - chord_length/2);
        x_upper = x_upper + shift_x;
        x_lower = x_lower + shift_x;
        
        % Shift so mid-chord is around y = Ly/2
        shift_y = Ly/2;
        y_upper = y_upper + shift_y;
        y_lower = y_lower + shift_y;
        
        % Merge upper and lower coordinates into one closed loop (CCW)
        contour_top    = [x_upper(:),  y_upper(:)];
        contour_bottom = [x_lower(:), y_lower(:)];
        contour_final  = [contour_top; contour_bottom];
    else
        error('Unrecognized shape name.');
    end
end

function [j_approx, i_approx] = nearest_grid_index(xx, yy, dx, dy)
%{
  Convert real (x,y) to integer grid indices (j,i).
  We'll assume j corresponds to y, i corresponds to x.
  We do a simple round. Then we clamp to valid array indices.

  In Python: i = round(xx/dx), j = round(yy/dy)
  In MATLAB (1-based): 
    i_approx = round(xx/dx) + 1 is the naive approach if x=0 => i=1
    but let's keep it simpler by ignoring that small shift. We'll do
    i_approx = round(xx/dx) + 1, j_approx = round(yy/dy) + 1
%}
    i_approx = round(xx/dx) + 1;
    j_approx = round(yy/dy) + 1;
end

function ds = panel_length(p1, p2, dx, dy)
%{
  Distance between two adjacent grid points (j1,i1) and (j2,i2),
  multiplied by dx and dy to get physical length. 
  p1 = [j1, i1], p2 = [j2, i2].
%}
    j1 = p1(1); i1 = p1(2);
    j2 = p2(1); i2 = p2(2);
    
    ds = sqrt( (i2 - i1)^2 * dx^2 + (j2 - j1)^2 * dy^2 );
end

function n = outward_normal(p1, p2)
%{
  Compute the outward normal from the segment p1->p2. 
  If we trace the body CCW, outward normal is found by rotating 
  the segment vector 90° LEFT: (dx, dy) -> (-dy, +dx).
  Then normalize.
%}
    j1 = p1(1); i1 = p1(2);
    j2 = p2(1); i2 = p2(2);

    dx_seg = (i2 - i1);
    dy_seg = (j2 - j1);

    n = [-dy_seg, dx_seg];
    Ln = norm(n);
    if Ln > 0
        n = n / Ln;
    end
end
