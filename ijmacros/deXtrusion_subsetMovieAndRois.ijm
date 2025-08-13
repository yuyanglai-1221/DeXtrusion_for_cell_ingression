/** 
 *  	DeXtrusion - subset movie
 *  	Save a temporal subset movie of original movie and its corresponding ROIs.
 *  	Keep the frames and the ROIs between two given time points.
 *  	The ROIs should have been saved by deXtrusion in the 'results' folder.
 *  	If keep_before is >0, keep few frames before the tstart point but not the ROIs, to allow visualisation of the events.
*/

resfold="/results/";  // where the ROI file is saved (default=results)
type = "cell_death";  // name of the event to show (default=extrusions)

// keep movie from t=tstart to t=tend
tstart = 180;
tend = 200;
// keep a few slices before the subset (only the images not the ROIs) to visualize the event corresponding to ROIs in the "tstart" slices.
// Put 0 not to keep slices before at all.
keep_before = 5;
keep_after = 5;   

srcimg = File.openDialog("Select the movie");
fold = File.getParent(srcimg);
imname = File.getName(srcimg);
imname = substring(imname, 0, lengthOf(imname)-4);

setBatchMode(true);
roiManager("reset");
open(srcimg);

tmin = tstart;
if (keep_before>0){tmin -= keep_before;}
if (tmin < 0 ){tmin = 1;}
tmaxi = tend;
if (keep_after>0){tmaxi += keep_after;}
run("Select None");
roiManager("deselect");
img = getImageID();

run("Duplicate...", "duplicate range="+tmin+"-"+tmaxi);
dup = getImageID();
save(fold+"/"+imname+"_subset_t"+tstart+"_t"+tend+".tif");

// get extrusions ROIs
roifile = fold+resfold+imname+"_"+type+".zip";
if (File.exists(roifile)){
	roiManager("Open", roifile);
	// Roi Manager options
	roiManager("Show All");
	roiManager("Sort");
	roiManager("Associate", "true");

	keepRois(tstart, tend, tmin, tmaxi);
}
selectImage(img);
close();
selectImage(dup);
roiManager("deselect");
roiManager("save", fold+resfold+imname+"_subset_t"+tstart+"_t"+tend+"_"+type+".zip" );
close();

setBatchMode(false);


function keepRois(tmin, tmax, tinit, tfinal)
{
	i = 0;
	while (i<roiManager("Count"))
	{
		selectImage(img);
		roiManager("Select", i);
		Roi.getPosition(chanel, slice, frame);
		if (slice > frame ) { frame = slice; }
		if ((tmin <= frame) & (frame<=tmax))
		{
			newpos = frame-tinit+1;
			selectImage(dup);
			roiManager("Select", i);
			setSlice(newpos);
			//Roi.setPosition(chanel, newpos, newpos);
			roiManager("Update");
			i = i + 1;
		} else {
			roiManager("Delete");
		}
	}
}



