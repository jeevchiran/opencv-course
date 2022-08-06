function [ H ] = est_homography(video_pts, logo_pts)
% est_homography estimates the homography to transform each of the
% video_pts into the logo_pts
% Inputs:
%     video_pts: a 4x2 matrix of corner points in the video
%     logo_pts: a 4x2 matrix of logo points that correspond to video_pts
% Outputs:
%     H: a 3x3 homography matrix such that logo_pts ~ H*video_pts
% Written for the University of Pennsylvania's Robotics:Perception course

% YOUR CODE HERE
H = [];
A = zeros(8,9);

% 4 pointts are provided hence 1 to 4
for i = 1 : 4
  % goal post coordinates to project banner
  x = video_pts(i,1);
  y = video_pts(i,2);

  % actual coordinates in banner
  x_prime = logo_pts(i,1)
  y_prime = logo_pts(i,2)

   % matrix H has 8 degrees of freedom, and so, as each point gives 2 sets of equations, we
   % need 4 points to solve for h uniquely. So, given four points (such as the corners provided
   % for this assignment), we can generate vectors ax and ay for each, and concatenate them:
   A(2 * i - 1, :) = [-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime];
   A(2 * i, :) = [0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime];
end

[U,S,V] = svd(A);
% The vector h will then be the last column of V , and you can then construct the 3x3
% homography matrix by reshaping the 9x1 h vector
H = V(:,end);
H = reshape(H,3,3);
H = H';
end

