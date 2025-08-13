/** 
 *  	DeXtrusion - check detected events
 *  	Show the ROIs on the original movie and ask for each if it's good or not.
 *  	The ROIs should have been saved by deXtrusion in the 'results' folder.
 *  	The corrected ROIs file will be saved in the same folder.
*/

resfold="/results/";  // where the ROI file is saved (default=results)
type = "cell_death";  // name of the event to show (default=extrusions)

srcimg = File.openDialog("Select the movie");
fold = File.getParent(srcimg);
imname = File.getName(srcimg);
imname = substring(imname, 0, lengthOf(imname)-4);
outname = imname+"_"+type+".zip";   // corrected ROIs file name: here, remplace the original file (same name)

winsize = 25;
nframes = 6;

roiManager("reset");
open(srcimg);

// get extrusions ROIs
roifile = fold+resfold+imname+"_"+type+".zip";
if (File.exists(roifile)){
	roiManager("Open", roifile);
	// Roi Manager options
	roiManager("Show All");
	roiManager("Sort");
	roiManager("Associate", "true");

	checkRois();
}

function checkRois()
{
	img = getImageID();
	i = 0;
	print("Initial nb of ROIs: "+roiManager("Count"));
	roiok = 0;
	roino = 0;
	while (i<roiManager("Count"))
	{
		selectImage(img);
		getDimensions(width, height, channels, slices, frames);
		nmax = maxOf( slices, frames );
		roiManager("Select", i);
		Roi.getPosition(chanel, slice, frame);
		Roi.getBounds(x, y, width, height)
		if (slice > frame ) { frame = slice; }
		setSlice(frame);
		makeRectangle(x-winsize, y-winsize, 2*winsize, 2*winsize);
		start_frame = maxOf(1, frame-nframes);
		end_frame = minOf( frame+nframes, nmax );
		run("Duplicate...", "duplicate range="+(start_frame)+"-"+(end_frame));
		dup = getImageID();
		if (nSlices>nframes) { setSlice(nframes); }
		run("Enhance Contrast", "saturated=0.35");
		getLocationAndSize(xst, yst, widthst, heightst);
		run("Make Montage...", "columns="+(nframes*2)+" rows=1 scale=4 border=2");
		montage = getImageID();
		setLocation(xst, yst+heightst*2);
		good = getBoolean("ROI "+i+"\n Correct event detected ? \n (Press \"y\" or \"n\" or click)");
		selectImage(dup);
		close();
		selectImage(montage);
		close();
		if (good)
		{
			roiok ++;
			i = i + 1;
		} else {
			roino ++;
			roiManager("Delete");
		}
	}
	print("Nb of correct ROIs: "+roiok);
	print("Nb of deleted ROIs: "+roino);
	print("Success rate (TP/TP+FP): "+(roiok/(roiok+roino)));
	roiManager("save", fold+resfold+outname );
}



