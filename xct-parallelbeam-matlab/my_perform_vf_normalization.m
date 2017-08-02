function vf = my_perform_vf_normalization(vf)

if size(vf,4)<=1
    d = sqrt( sum(vf.^2,3) );
else
    d = sqrt( sum(vf.^2,4) );
end
d(d<1e-9) = 1;
vf = prod_vf_sf(vf,1./d);