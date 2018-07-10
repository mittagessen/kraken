// load image to canvas

window.onload = function() {
    // Mouse tracking to image
    document.getElementById("page-image").addEventListener("mouseenter",
                                                           function() {
                                                               inDiv=true;
                                                           });
    document.getElementById("page-image").addEventListener("mouseout",
                                                           function() {
                                                               inDiv=false;
                                                           });
    document.getElementById("page-image").addEventListener(
        "dblclick",
        function (){
        }
    );
    // End of mouse tracking
    //
    // Canvas load image
    var container = document.getElementById("page-image");
    var canvas = document.getElementById("image-canvas");
    console.log(canvas);
    var context = canvas.getContext('2d');
    console.log(context);
    var image = new Image()
    image.src = imgPath;
    canvas.style.width = "100%";
    // var h = 3 * canvas.width / 4
    // canvas.clientHeight = h;
    // canvas.style.height = "100%";
    var imratio = image.height / image.width;
    var aspectR = 1.5;
    var cwidth = canvas.clientWidth;
    canvas.height = cwidth * imratio;
    var cheight = canvas.clientHeight;
    var hRatio = canvas.width / image.width;
    var vRatio = canvas.height / image.height;
    var cvRatio = image.height / canvas.height;
    var ratio  = Math.min( hRatio, vRatio );
    var centerShift_x = ( canvas.width - image.width*ratio ) / 2;
    var centerShift_y = ( canvas.height - image.height*ratio ) / 2;
    console.log("canvas width");
    console.log(cwidth);
    console.log("canvas height");
    console.log(cheight);
    image.onload = function(){
        context.drawImage(image,
                          0,0,// coordinate source
                          image.width, // source rectangle width
                          image.height, // source rectangle height
                          centerShift_x, centerShift_y, // destination coordinate
                          // 0,0,
                          image.width*ratio, // destination width
                          image.height*ratio // destination height
                         );
    };
    console.log("now image");
    console.log(image);
    console.log("image width");
    console.log(image.clientWidth);
    console.log("image height");
    console.log(image.clientHeight);
    console.log("canvas width");
    console.log(canvas.clientWidth);
    // image.clientWidth = canvas.clientWidth;
    //
    var i=0;
    context.beginPath();
    context.lineWidth = 2;
    context.strokeStyle = "red";
    // while(i < lines.length){
    //     var lineObj = lines[i];
    //     var leftInt = parseInt(lineObj.left, 10);
    //     leftInt = Math.floor(leftInt);
    //     var topInt = parseInt(lineObj.top, 10);
    //     topInt = Math.floor(topInt);
    //     var widthInt = parseInt(lineObj.width, 10);
    //     widthInt = Math.floor(widthInt);
    //     var heightInt = parseInt(lineObj.height, 10);
    //     heightInt = Math.floor(heightInt);
    //     context.rect(leftInt, topInt, widthInt, heightInt);
    //     context.stroke();
    //     i += 1;
    // }
};


// sort function for lists
var deletedNodes = [];
function deleteBoxes(){
    /*
      Simple function for deleting lines whose checkboxes are selected
      Description:
      We query the input elements whose class is delete-checkbox.
      Then we check whether they are checked or not.
      If they are checked we delete the item group containing them
    */
    var deleteCheckBoxList = document.querySelectorAll("input.delete-checkbox");
    var dellength = deleteCheckBoxList.length;
    var deletedboxlength = 0;
    var i = 0;
    while(i < dellength){
        var cbox = deleteCheckBoxList[i];
        var checkval = cbox.checked;
        if(checkval===true){
            // removing the element if checkbox is checked
            var labelnode = cbox.parentNode;
            var linewidgetNode = labelnode.parentNode;
            var itemlistNode = linewidgetNode.parentNode;
            var itemgroupNode = itemlistNode.parentNode;
            var lineId = itemgroupNode.id;
            // get the image line from the other column
            var imageLineSelector = "a.rect[id='".concat(lineId,
                                                         "']");
            var imageLine = document.querySelector(imageLineSelector);
            //
            var imageparent = imageLine.parentNode;
            var itemparent = itemgroupNode.parentNode;
            var deleted = {"imageline" : imageLine,
                           "itemgroup" : itemgroupNode,
                           "imageparent" : imageparent,
                           "itemparent" : itemparent};
            deletedNodes.push(deleted);
            // remove both from the page
            itemparent.removeChild(itemgroupNode);
            imageparent.removeChild(imageLine);
            deletedboxlength +=1;
        }
        i+=1;
    }
    if(deletedboxlength === 0){
        alert("Please select lines for deletion");
    }
}
//
function undoDeletion(){
    if (deletedNodes.length === 0){
        alert("Deleted line information is not found");
    }
    var lastObject = deletedNodes.pop();
    //
    var imageLine = lastObject["imageline"];
    var itemgroup = lastObject["itemgroup"];
    var imageparent = lastObject["imageparent"];
    var itemparent = lastObject["itemparent"];
    //
    imageparent.appendChild(imageLine);
    itemparent.appendChild(itemgroup);
    //
}
function sortLines() {
    var lineList = document.querySelectorAll(".item-group");
    var itemparent = document.querySelector("#text-line-list");
    var linearr = Array.from(lineList).sort(
        (a,b) => parseInt(a.id, 10) - parseInt(b.id, 10)
    );
    linearr.forEach(el => itemparent.appendChild(el) );
}

function getMousePosition(event) {
    var posX = event.clientX;
    var posY = event.clientY;
    //
    return posX, posY;
}
// add mouse tracking to page image

var inDiv = false;


function getMousePositionImage(event) {
    var result;
    if(inDiv === false){
        result = null;
    }else{
        result = getMousePosition(event);
    }
    return result;
}
