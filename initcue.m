function cue = initcue(s,kernel_type)
cue.kernel.type = kernel_type;
cue.features.gray = false;
cue.features.hog = false;
switch s
    case 'gray',
        cue.interp_factor = 0.02;  %linear interpolation factor for adaptation
		cue.kernel.sigma = 0.2;  %gaussian kernel bandwidth
		cue.kernel.poly_a = 1;  %polynomial kernel additive term
		cue.kernel.poly_b = 7;  %polynomial kernel exponent
		cue.features.gray = true;
		cue.cell_size = 1;
    case 'hog',
        cue.interp_factor = 0.02;
		cue.kernel.sigma = 0.5;
		cue.kernel.poly_a = 1;
		cue.kernel.poly_b = 9;
		cue.features.hog = true;
		cue.features.hog_orientations = 9;
		cue.cell_size = 4;
    otherwise
        error('Unknown cue.')
end
