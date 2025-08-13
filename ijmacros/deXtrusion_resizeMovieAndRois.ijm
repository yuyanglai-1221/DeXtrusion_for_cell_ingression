/** 
 *  	DeXtrusion - resize movie and ROI
 *  	Resize spatially and temporally a movie with its corresponding ROIs
 *  	ROIs should be named name_of_the_movie+"_cell_death.zip" (or other event)
 *  	Usefull to scale the movies to train/retrain a DeXNet
*/

srcimg = File.openDialog("Select the movie");
fold = File.getParent(srcimg);
imname = File.getName(srcimg);
imname = substring(imname, 0, lengthOf(imname)-4);
open(fold+"/"+imname+".tif");
	
outfold=fold+"/rescaled/";  // where the rescaled files are saved 

event = "cell_death"; // which ROI to resize
orig_cell_diam = 80;  // typical cell diameter in the original movie
orig_extr_duration = 2;  // typical duration of cell extrusion in the original movie

target_cell_diam = 25;
target_extr_duration = 4.5;

fact = target_cell_diam/orig_cell_diam;
facttemp = target_extr_duration/orig_extr_duration;

getDimensions(width, height, channels, slices, frames);
// check if image is in time (frames) or z (slices)
// put it to time mode (frames)
nimg = channels+slices+frames-2;
astime = 1;
if (slices>frames) {
	astime = 0;
	run("Properties...", "channels=1 slices=1 frames="+nimg+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
}

// Resize the image in a new image
orig = getImageID();
run("Duplicate...", "ignore duplicate");
run("Size...", "width="+(getWidth()*fact)+" depth="+Math.ceil(nimg*facttemp)+" constrain interpolation=Bilinear");
resized = getImageID();
selectImage(orig);
// event to rescale
resizeRois(event);
run("Hide Overlay");
selectImage(resized);
saveAs("Tif", outfold+imname+".tif");
close();
selectImage(orig);
close();

function resizeRois(event)
{
	run("Hide Overlay");
	roiManager("reset");
	roiManager("Open", fold+"/"+imname+"_"+event+".zip");
	
	nroi = roiManager("count");
	i = 0;
	while ( i<nroi) {
		selectImage(orig);
		roiManager("Select", 0);
		Roi.getCoordinates(x, y);
		Roi.getPosition(channel, slice, frame);
		roiManager("delete");
		if (frame>slice){
			slice = frame;
			setSlice(slice);
		}else {
			setSlice(slice);
		}
		newx = x[0]*fact;
		if (newx<=0){newx =1;}
		newy = y[0]*fact;
		if (newy<=0){newy =1;}
		
		newframe=Math.ceil(slice*facttemp);
		if (newframe==0){newframe=1;}
		
		selectImage(resized);
		setSlice(newframe);
		makePoint(newx, newy, "extra large red circle");
		Roi.setPosition(1,1,newframe);
		
		roiManager("add");
		i = i+1;	
	}
	roiManager("deselect");
	roiManager("Save", outfold+imname+"_"+event+".zip");
}

