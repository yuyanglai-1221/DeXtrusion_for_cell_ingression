/** 
 *  	DeXtrusion - overlay probability map
 *  	Show the probability maps on the original movie 
 *  	The maps should have been saved by deXtrusion in the 'results' folder.
 *  	Open all the maps and overlay it on the original movie as a Composite movie
*/

srcimg = File.openDialog("Select the movie");
fold = File.getParent(srcimg);
imname = File.getName(srcimg);
imname = substring(imname, 0, lengthOf(imname)-4);
resfold="/results/";

open(srcimg);
run("8-bit");
getDimensions(imwidth, imheight, imchannels, imslices, imframes);
nimg = imchannels+imslices+imframes-2;
run("Properties...", "channels=1 slices=1 frames="+nimg+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
	
		
// get extrusion map 
deathfile = fold+resfold+imname+"_cell_death_proba.tif";
death = 0;
if (File.exists(deathfile)){
	open(deathfile);
	getDimensions(width, height, channels, slices, frames);
	nimg = channels+slices+frames-2;
	run("Properties...", "channels=1 slices=1 frames="+nimg+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
	death = getImageID();
	if ((width+height) != (imwidth+imheight))
	{
		run("Size...", "width="+imwidth+" height="+imheight+" interpolation=Bilinear");
	}
	deathtitle = getTitle();
	setMinAndMax(150,255);  // above other events
}

// get sop map
sopfile = fold+resfold+imname+"_cell_sop_proba.tif";
sop = 0;
if (File.exists(sopfile)) {
	open(sopfile);
	getDimensions(width, height, channels, slices, frames);
	nimg = channels+slices+frames-2;
	run("Properties...", "channels=1 slices=1 frames="+nimg+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
	sop = getImageID();
	if ((width+height) != (imwidth+imheight))
	{
		run("Size...", "width="+imwidth+" height="+imheight+" interpolation=Bilinear");
	}
	soptitle = getTitle();
	setMinAndMax(150,255);  // above other events
}

// get division map
divfile = fold+resfold+imname+"_cell_division_proba.tif";
div = 0;
if (File.exists(divfile)) {
	open(divfile);
	getDimensions(width, height, channels, slices, frames);
	nimg = channels+slices+frames-2;
	run("Properties...", "channels=1 slices=1 frames="+nimg+" pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
	div = getImageID();
	if ((width+height) != (imwidth+imheight))
	{
		run("Size...", "width="+imwidth+" height="+imheight+" interpolation=Bilinear");
	}
	divtitle = getTitle();
	setMinAndMax(150,255);  // above other events
}

if (death!=0&& sop!=0 && div !=0){
   run("Merge Channels...", "c1="+deathtitle+" c2="+divtitle+" c3="+soptitle+" c4="+imname+".tif create");
} else {
	if (death!=0&& sop!=0){
	run("Merge Channels...", "c1="+deathtitle+" c3="+soptitle+" c4="+imname+".tif create");
	} else {
		if (death!=0){
		run("Merge Channels...", "c1="+deathtitle+" c4="+imname+".tif create");
		}
	}
}	

