load iter0_pca__oscillatory_IC3.pdb, traj
split_states traj, 1, 1
split_states traj, 2, 2
split_states traj, 3, 3
split_states traj, 4, 4
split_states traj, 5, 5
split_states traj, 6, 6
split_states traj, 7, 7
split_states traj, 8, 8
split_states traj, 9, 9
split_states traj, 10, 10
set orthoscopic
bg_color white
set specular, off
set ribbon_width, 0.5
set ribbon_trace_atoms
hide nonbonded
as ribbon
util.cbc
orient
zoom center, 150
turn z, 90
hide
show ribbon, traj_0001
ray 1200,1200
png IC3_movie/movie_0001.png
hide
show ribbon, traj_0002
ray 1200,1200
png IC3_movie/movie_0002.png
hide
show ribbon, traj_0003
ray 1200,1200
png IC3_movie/movie_0003.png
hide
show ribbon, traj_0004
ray 1200,1200
png IC3_movie/movie_0004.png
hide
show ribbon, traj_0005
ray 1200,1200
png IC3_movie/movie_0005.png
hide
show ribbon, traj_0006
ray 1200,1200
png IC3_movie/movie_0006.png
hide
show ribbon, traj_0007
ray 1200,1200
png IC3_movie/movie_0007.png
hide
show ribbon, traj_0008
ray 1200,1200
png IC3_movie/movie_0008.png
hide
show ribbon, traj_0009
ray 1200,1200
png IC3_movie/movie_0009.png
hide
show ribbon, traj_0010
ray 1200,1200
png IC3_movie/movie_0010.png
