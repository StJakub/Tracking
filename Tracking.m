function Tracking

global pattern videoFileReader

% Prog dokladnosci
threshold = single(0.95);

video_img = step(videoFileReader);
video_img = rgb2gray(video_img);
video_img = single(video_img);
pattern = single(pattern);

%Redukcja, w celu zmniejszenia obliczen
video_img =  impyramid(video_img, 'reduce');
pattern_down = impyramid(pattern, 'reduce');


pattern_energy = sqrt(sum(pattern_down(:).^2));
pattern_down_rot = imrotate(pattern_down, 180);
[r_pattern, c_pattern] = size(pattern_down_rot);
[r_video, c_video]= size(video_img);

%Zmiana rozmiaru pod fft2
rows2fft = 2^nextpow2(r_video + r_pattern);
col2fft = 2^nextpow2(c_pattern + c_video);
pattern_cor = [pattern_down_rot zeros(r_pattern, col2fft-c_pattern)];
pattern_cor = [pattern_cor; zeros(rows2fft-r_pattern, col2fft)];

pattern_fft = fft2(pattern_cor);

target_size = size(pattern);
Im_video = zeros(rows2fft, col2fft, 'single'); 
C_ones = ones(r_pattern, c_pattern, 'single'); 

LocalMaxFinder = vision.LocalMaximaFinder( 'Threshold', single(-1), 'MaximumNumLocalMaxima', 1);      
Player = vision.VideoPlayer('Name', 'Detekcja');


while ~isDone(videoFileReader)
    Image = step(videoFileReader);
    Image_color = Image;
    Image_gray = rgb2gray(Image);
    
    Image_down = impyramid(Image_gray, 'reduce');
  
    Im_video(1:r_video, 1:c_video) = Image_down;   
    image_fft = fft2(Im_video);
    %Korelacja
    corr_frequency = image_fft .* pattern_fft;
    corr_classic = ifft2(corr_frequency);
    corr_classic = corr_classic(r_pattern:r_video, c_pattern:c_video);
    
    Image_energy = (Image_down).^2;
    Image_c = conv2(Image_energy, C_ones, 'valid');
    Image_c = sqrt(Image_c);

    %Normalizacja
    Corr_normalized = (corr_classic) ./ (Image_c * pattern_energy);
    xyLocation = step(LocalMaxFinder, Corr_normalized);
    
    %Lokalizacja
    coordinates = sub2ind([r_video - r_pattern, c_video-c_pattern]+1, xyLocation(:,2),xyLocation(:,1));

    Corr_column = Corr_normalized(:);
    norm_Corr_value = Corr_column(coordinates);
    detect = (norm_Corr_value > threshold);
    pattern_roi = zeros(length(detect), 4);
    ul_corner = (2.*(xyLocation(detect, :)-1))+1;
    pattern_roi(detect, :) = [ul_corner, fliplr(target_size(detect, :))];
    
    %Zaznaczanie
    Roi_draw = insertShape(Image_color, 'Rectangle', pattern_roi, 'Color', 'green');
    
    step(Player, Roi_draw);
end

snapnow
release(videoFileReader);

end