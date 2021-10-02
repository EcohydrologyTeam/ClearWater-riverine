#!/bin/zsh

# Convert png files to jpeg files
# for i in muncie_chloropleth*.png; do sips -s format jpeg -s formatOptions 70 "${i}" --out "${i%png}jpg"; done

# Convert jpg files to animated GIF using ImageMagick's convert command
# convert -delay 60 -loop 0 -size 640x480 muncie_chloropleth*.jpg muncie_chloropleth_maps.gif
# convert -delay 60 -loop 0 -resize 1800x1350 muncie_chloropleth_045.jpg muncie_chloropleth_046.jpg muncie_chloropleth_047.jpg muncie_chloropleth_maps.gif
convert -delay 60 -loop 0 -size 1800x1350 muncie_chloropleth*.jpg muncie_chloropleth_maps.gif
