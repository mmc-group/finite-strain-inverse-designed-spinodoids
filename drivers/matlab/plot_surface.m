function plot_surface(X,Y,Z,S, enable_isocaps)
    figure
    [f,v,c]=getSurface(X,Y,Z,S,0);
    if enable_isocaps
        [fc,vc,cc]=getCaps(X,Y,Z,S,0);
        [f,v,c]=joinElementSets({f,fc},{v,vc},{c,cc}); %Join sets
    end
    gpatch(f,v,c,'none');
    axisGeom; camlight headlight;
    colormap gjet; icolorbar;
    gdrawnow;
%     pause(0.5);
%     close;
end

function [f,v,c]=getSurface(X,Y,Z,S,levelset)
    [f,v] = isosurface(X,Y,Z,S,levelset);
    c=zeros(size(f,1),1);
end

function [fc,vc,cc]=getCaps(X,Y,Z,S,levelset)
    [fc,vc] = isocaps(X,Y,Z,S,levelset,'enclose','below');     %Compute isocaps
    
    nc=patchNormal(fc,vc);
    cc=zeros(size(fc,1),1);
    cc(nc(:,1)<-0.5)=1;
    cc(nc(:,1)>0.5)=2;
    cc(nc(:,2)<-0.5)=3;
    cc(nc(:,2)>0.5)=4;
    cc(nc(:,3)<-0.5)=5;
    cc(nc(:,3)>0.5)=6;    
end