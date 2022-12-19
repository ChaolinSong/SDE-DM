function  srgtSRGT = Trainrbf(sample_x,sample_y)
srgtOPT  = srgtsRBFSetOptions(sample_x, sample_y,@rbf_build,[],'CU',0.1,0);
srgtSRGT = srgtsRBFFit(srgtOPT);
end