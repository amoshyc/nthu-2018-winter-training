function [res] = lbp(img)
    [m, n] = size(img);
    
    img0 = img(1:end-2, 1:end-2);
    img1 = img(1:end-2, 2:end-1);
    img2 = img(1:end-2, 3:end);
    img3 = img(2:end-1, 3:end);
    img4 = img(3:end, 3:end);
    img5 = img(3:end, 2:end-1);
    img6 = img(3:end, 1:end-2);
    img7 = img(2:end-1, 1:end-2);
    img = img(2:end-1, 2:end-1);
    
    d0 = uint8(img >= img0);
    d1 = uint8(img >= img1);
    d2 = uint8(img >= img2);
    d3 = uint8(img >= img3);
    d4 = uint8(img >= img4);
    d5 = uint8(img >= img5);
    d6 = uint8(img >= img6);
    d7 = uint8(img >= img7);
    
    e0 = ones(m - 2, n - 2, 'uint8') .* 1;
    e1 = ones(m - 2, n - 2, 'uint8') .* 2;
    e2 = ones(m - 2, n - 2, 'uint8') .* 4;
    e3 = ones(m - 2, n - 2, 'uint8') .* 8;
    e4 = ones(m - 2, n - 2, 'uint8') .* 16;
    e5 = ones(m - 2, n - 2, 'uint8') .* 32;
    e6 = ones(m - 2, n - 2, 'uint8') .* 64;
    e7 = ones(m - 2, n - 2, 'uint8') .* 128;
    
    c0 = (d0 .* e0) + (d1 .* e1) + (d2 .* e2) + (d3 .* e3) + ...
         (d4 .* e4) + (d5 .* e5) + (d6 .* e6) + (d7 .* e7);
    c1 = (d0 .* e1) + (d1 .* e2) + (d2 .* e3) + (d3 .* e4) + ...
         (d4 .* e5) + (d5 .* e6) + (d6 .* e7) + (d7 .* e0);
    c2 = (d0 .* e2) + (d1 .* e3) + (d2 .* e4) + (d3 .* e5) + ...
         (d4 .* e6) + (d5 .* e7) + (d6 .* e0) + (d7 .* e1);
    c3 = (d0 .* e3) + (d1 .* e4) + (d2 .* e5) + (d3 .* e6) + ...
         (d4 .* e7) + (d5 .* e0) + (d6 .* e1) + (d7 .* e2);
    c4 = (d0 .* e4) + (d1 .* e5) + (d2 .* e6) + (d3 .* e7) + ...
         (d4 .* e0) + (d5 .* e1) + (d6 .* e2) + (d7 .* e3);
    c5 = (d0 .* e5) + (d1 .* e6) + (d2 .* e7) + (d3 .* e0) + ...
         (d4 .* e1) + (d5 .* e2) + (d6 .* e3) + (d7 .* e4);
    c6 = (d0 .* e6) + (d1 .* e7) + (d2 .* e0) + (d3 .* e1) + ...
         (d4 .* e2) + (d5 .* e3) + (d6 .* e4) + (d7 .* e5);
    c7 = (d0 .* e7) + (d1 .* e0) + (d2 .* e1) + (d3 .* e2) + ...
         (d4 .* e3) + (d5 .* e4) + (d6 .* e5) + (d7 .* e6);
    
    mn0123 = min(min(c0, c1), min(c2, c3));
    mn4567 = min(min(c4, c5), min(c6, c7));
    res = histogram(min(mn0123, mn4567), 0:256);
    disp(res.Values);
end


