function [res] = lbp2(img)
    [m, n] = size(img);
    res = ones(m, n, 'uint8') * 255;
    dr = [-1 -1 -1 0 +1 +1 +1 0];
    dc = [-1 0 +1 +1 +1 0 -1 -1];
    e = [1 2 4 8 16 32 64 128];
    
    for r = 2:(m-1)
        for c = 2:(n-1)
            neighbors = zeros(8, 1);
            for d = 1:8
                if img(r, c) >= img(r + dr(d), c + dc(d))
                    neighbors(d) = 1;
                end
            end
            
            for d = 1:8
                val = dot(neighbors, e);
                res(r, c) = min(res(r, c), val);
                neighbors = circshift(neighbors, 1);
            end
        end
    end
        
    res = histogram(res(2:end-1, 2:end-1), 0:256);
end