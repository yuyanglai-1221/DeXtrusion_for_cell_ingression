/** 
 *  	DeXtrusion - overlay probability map
 *  	Show the ROIs on the original movie 
 *  	The ROIs should have been saved by deXtrusion in the 'results' folder.
*/

resfold="/results/";  // where the ROI file is saved (default=results)
type = "cell_death";  // name of the event to show (default=extrusions)

srcimg = File.openDialog("Select the movie");
fold = File.getParent(srcimg);
imname = File.getName(srcimg);
imname = substring(imname, 0, lengthOf(imname)-4);

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
}



