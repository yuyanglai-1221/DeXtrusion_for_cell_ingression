/** 
 *  	DeXtrusion - score detected events
 *  	Show randomly selected ROIs on the original movie and ask for each if it's good or not.
 *  	The ROIs should have been saved by deXtrusion in the 'results' folder.
 *  	Allows to quantify the quality of the results.
*/

resfold="/results/";  // where the ROI file is saved (default=results)
type = "cell_death";  // name of the event to show (default=extrusions)

checkrois = 100;  // number of ROIs to check

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
	print("Total nb of ROIs: "+roiManager("Count"));
	roiok = 0;
	roino = 0;
	doneroi = 0;
	while ( (doneroi<checkrois) && (roiManager("count")>0) )
	{
		selectImage(img);
		i = floor(random()*roiManager("count"));
		roiManager("Select", i);
		Roi.getPosition(chanel, slice, frame);
		Roi.getBounds(x, y, width, height)
		if (slice > frame ) { frame = slice; }
		setSlice(frame);
		makeRectangle(x-winsize, y-winsize, 2*winsize, 2*winsize);
		run("Duplicate...", "duplicate range="+(frame-nframes)+"-"+(frame+nframes));
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
			roiManager("Delete");
		} else {
			roino ++;
			roiManager("Delete");
		}
		doneroi = doneroi + 1;
		print(doneroi+" "+roiok+" "+roino+" "+(roiok/(roiok+roino)));
	}
	print("Nb of correct ROIs: "+roiok);
	print("Nb of deleted ROIs: "+roino);
	print("Success rate (TP/TP+FP): "+(roiok/(roiok+roino)));
}



