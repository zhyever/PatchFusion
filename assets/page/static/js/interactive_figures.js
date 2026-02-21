

// // Function to update the clip size for a given top image and slider value
function setClipSize(topImage, value) {
    var imgWidth = topImage.offsetWidth;
    var clipValue = (value / 100) * imgWidth;
    // If using 'clip-path', the syntax might differ
    topImage.style.clip = `rect(0px, ${clipValue}px, ${topImage.offsetHeight}px, 0px)`;
}

// Attach event listeners to all sliders
document.querySelectorAll('.slider').forEach(function(slider) {
    // Initial setting
    setClipSize(slider.previousElementSibling, slider.value);
    
    // Update on input
    slider.addEventListener('input', function() {
        setClipSize(this.previousElementSibling, this.value);
    });
});

// If you need to handle window resizing (if the image size is responsive), you may also want to add:
window.addEventListener('resize', function() {
    document.querySelectorAll('.slider').forEach(function(slider) {
        setClipSize(slider.previousElementSibling, slider.value);
    });
});



