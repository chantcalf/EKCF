function results=run_EKCF(seq, res_path, bSaveImage)
%size = 4.5

close all

feature_type = 'hogcolor';
kernel_type = 'linear';	
features.gray = false;
features.hog = false;
features.hogcolor = false;
kernel.type = kernel_type;

padding = 1.5;  %extra area surrounding the target
paddings = 10;
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
miu = 5e-5;
tt = 1;

SampleSelect = true;
TargetEnhancement = true;

switch feature_type
	case 'gray',
		interp_factor = 0.075;  %linear interpolation factor for adaptation
		kernel.sigma = 0.2;  %gaussian kernel bandwidth
		
		kernel.poly_a = 1;  %polynomial kernel additive term
		kernel.poly_b = 7;  %polynomial kernel exponent
	
		features.gray = true;
		cell_size = 1;
	case 'hog',
		interp_factor = 0.02;
		
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
    case 'hogcolor',
		interp_factor = 0.01;
		
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hogcolor = true;
		features.hog_orientations = 9;
		cell_size = 4;	
    otherwise
    error('Unknown feature.')
end

temp = load('w2crs');
w2c = temp.w2crs;

show_visualization = true;
%show_visualization = false;

video_path = '';
img_files = seq.s_frames;
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);
resize_s = 0;
resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
        resize_s = 0.5;
    else
        if target_sz(1)<50 && target_sz(2)<50,
            resize_image = 1;
            pos = floor(pos * 2);
            target_sz = floor(target_sz * 2);
            resize_s = 2;
        end
    end


    %window_sz = floor(target_sz * (1 + padding));
	d = floor((-1*(target_sz(1)+target_sz(2))+sqrt((target_sz(1)+target_sz(2))^2+4*(paddings)*target_sz(1)*target_sz(2)))/2);
    window_sz = target_sz + [d,d];

	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
    y_sz = floor(window_sz / cell_size);
    y = gaussian_shaped_labels(output_sigma,y_sz);
    yf = fft2(y);
    
    t_sz = floor(target_sz/cell_size);
    q = zeros(y_sz);
    q1 = q;
    %
    sd = 3;
    s0 = floor((y_sz-sd*t_sz)/2)+1;
    s1 = s0 + floor(sd*t_sz);
    s0(s0<1) = 1;
    if s1(1)>y_sz(1)
        s1(1) = y_sz(1);
    end
    if s1(2)>y_sz(2)
        s1(2) = y_sz(2);
    end
    q1(s0(1):s1(1), s0(2):s1(2)) = 1;
    %}
    q2 = q;
    dd = d;
    r_sz = floor(dd/cell_size);
    ss = floor((y_sz-r_sz)/2);
    q2(ss(1)+1:ss(1)+r_sz,ss(2)+1:ss(2)+r_sz) = 1;
    q3 = q1+q2;
    q(q3 == 2) = 1;
    %imshow(q3/2.0);return
    q = 1-q;
    q = circshift(q, -floor(y_sz(1:2) / 2) + 1);
    
    %y = y.*(1-q);
    %yf = fft2(y);

    cos_window2 = ones(y_sz);
    cos_window = hann(y_sz(1)) * hann(y_sz(2))';
    if SampleSelect && TargetEnhancement
        cos_window = cos_window.^0.5;
    end
    cos_a = 15*target_sz./window_sz;
    t_window = gausswin(y_sz(1),cos_a(2)) * gausswin(y_sz(2),cos_a(1))';
    t_window(t_window<0.01) = 0;
    
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_s);
    end
    scale_level = [1 0.995 1.005 0.99  1.01 0.985 1.015];
    ns = 7;
    scale = 1;
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    target_sz0 = target_sz;
	for frame = 1:numel(img_files),
		%load image
        %frame
		im = imread([video_path img_files{frame}]);
        
        iim=im;
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, resize_s);
            iim = imresize(iim, resize_s);
        end
        
                
		tic()
        
		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
            %patch = get_subwindow(im, pos, window_sz);
            %patch1 = get_subwindow(iim, pos, window_sz);
            pp0 = 0;
            for i = 1:ns
                scales = scale*scale_level(i);
                patch0 = get_subwindow(iim, pos, floor(window_sz*scales));
                patch = imresize(patch0,window_sz);
                z = get_features(patch, features, cell_size, cos_window2,w2c);

                z = bsxfun(@times, z, cos_window);
                zf = fft2(z);
                kzf = kernel_correlation(zf,model_xf,kernel);

                response0 = real(ifft2(kzf.*model_alphaf));

                pp=max(response0(:));
                if i==1 || pp>pp0
                    pp0 = pp;
                    response = response0;
                    nsi = i;
                end
            end

            scale = scale*scale_level(nsi);

			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
			if vert_delta > y_sz(1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - y_sz(1);
			end
			if horiz_delta > y_sz(2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - y_sz(2);
            end
            
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1]*scale;
            target_sz = target_sz0*scale;
		end

		%obtain a subwindow for training at newly estimated target position
        if mod(frame,1)==0
        patch0= get_subwindow(iim, pos, floor(window_sz*scale));
        patch = imresize(patch0,window_sz);
        
        x = get_features(patch, features, cell_size,cos_window,w2c);
        xf = fft2(x);
        if TargetEnhancement
            x0 = bsxfun(@times, x, t_window);
        else
            x0 = x;
        end
        x0f = fft2(x0);
        axf = x0f;
        kf = kernel_correlation(x0f,x0f,kernel);
        alphaf =yf ./ (kf + lambda);%equation for fast training
        
        if SampleSelect
            delta = 2;
            ite = 0;
            ss = size(alphaf,1)*size(alphaf,2);
            while delta>0.02
                alpha = real(ifft2(alphaf));
                thb = tt*q/2/miu;
                belta = abs(alpha) - thb;
                belta(belta<0) = 0;
                belta = sign(alpha).*belta;
                beltaf = fft2(belta);
                alphaf = (kf.*yf+miu*beltaf)./(kf.*kf+lambda*kf+miu);
                deltam = abs(real(ifft2(alphaf))-alpha);
                delta = sum(deltam(:))/ss;
                ite = ite+1;
                if ite>20 break;end;
                %ite
                %delta
            end
        end
        
		if frame == 1,  %first frame, train with a single image
            model_alphaf = alphaf;
			model_xf = axf;
		else
			%subsequent frames, interpolate model
            model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
            model_xf = (1 - interp_factor) * model_xf + interp_factor * axf;
        end
        end

		%save position and timing
		positions(frame,:) = pos;
        rect = [pos-target_sz/2,target_sz];
        rect = [rect(2),rect(1),rect(4),rect(3)];
        res(frame,:) = rect;
		time = time + toc();

		%visualization
		if show_visualization,
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
        end
        
        
	end

	if resize_image,
		positions = positions /resize_s;
        res = res /resize_s;
	end



fps = numel(img_files) / time;

disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = res;%each row is a rectangle
results.fps = fps;

%show the precisions plot
% show_precision(positions, ground_truth, video_path)
end


function kf = kernel_correlation(xf,yf,kernel)
    switch kernel.type
    case 'gaussian',
        kf = gaussian_correlation(xf, yf, kernel.sigma);
    case 'polynomial',
        kf = polynomial_correlation(xf, yf, kernel.poly_a, kernel.poly_b);
    case 'linear',
        kf = linear_correlation(xf, yf);
    end
end
