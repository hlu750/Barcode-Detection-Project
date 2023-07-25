# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):     
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)     
    for i in range(image_height):                 
        for j in range(image_width):                       
            grey = 0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j]             
            greyscale_pixel_array[i][j] = round(grey)   
    return greyscale_pixel_array

def computeMinAndMaxValues(pixel_array, image_width, image_height):
    minimum = pixel_array[0][0]
    maximum = minimum

    for row in pixel_array:
        if min(row) < minimum:
            minimum = min(row)
        if max(row) > maximum:
            maximum = max(row)
    return (minimum, maximum)
    
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    new = createInitializedGreyscalePixelArray(image_width, image_height)

    min, max = computeMinAndMaxValues(pixel_array, image_width, image_height)

    if max == min:
        return new

    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[0])):
            new[i][j] = round((pixel_array[i][j]-min)/(max-min)*255)

    return new

def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    vertical = createInitializedGreyscalePixelArray(image_width, image_height,0.0)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            acc = pixel_array[i-1][1+j] + (pixel_array[i][1 + j]) * 2 + pixel_array[1 + i][1 + j] - pixel_array[i - 1][j - 1] - (pixel_array[i][j - 1]) * 2 - pixel_array[1 + i][j - 1]

            vertical[i][j] = abs(float(acc/8.0))

    return vertical

def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    horizontal = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)
    
    for j in range(1, image_height - 1):
        for i in range(1, image_width - 1):
            acc = (pixel_array[j - 1][i - 1] + (pixel_array[j - 1][i])* 2 + pixel_array[j - 1][1 + i] - pixel_array[1 + j][i - 1] - (pixel_array[1 + j][i]) * 2 - pixel_array[1 + j][1 + i])
            acc = abs(acc / 8.0)
            horizontal[j][i] = acc
            
    return horizontal

def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    mean = createInitializedGreyscalePixelArray(image_width, image_height,0.0)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            count = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    count += pixel_array[i + a][j + b]
            mean[i][j] = count / 9.0

    return mean

def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height): 
    result = [[0] * image_width for i in range(image_height)]  
    kernel = [ [1, 2, 1],                
            [2, 4, 2],                
            [1, 2, 1] ] 
    scale = 1 / 14 
    for row in kernel:         
        for i in range(len(row)):             
            row[i] *= scale    
    offsetI = -(len(kernel) // 2)     
    offsetJ = -(len(kernel[0]) // 2)    
    for i in range(image_height):         
        for j in range(image_width):                       
            for ki in range(len(kernel)):                 
                for kj in range(len(kernel[ki])):                         
                    i_ = i + ki + offsetI                     
                    j_ = j + kj + offsetJ                            
                    if i_ < 0:                         
                        i_ = 0                     
                    elif i_ >= image_height:                         
                        i_ = image_height - 1                     
                    if j_ < 0:                         
                        j_ = 0                     
                    elif j_ >= image_width:                         
                        j_ = image_width - 1                     
                    result[i][j] += pixel_array[i_][j_] * kernel[ki][kj] 
    return result  

def computeStandardDeviationImage3x3(pixel_array, image_width, image_height): 
    image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0) 
    
    for i in range(1, image_height-1):                
        for j in range(1, image_width-1):             
            mean = 0                                    
            for k in range(i-1, i+2):                 
                for l in range(j-1, j+2):                     
                    mean += pixel_array[k][l]               
            mean /= 9                                
            var = 0                                    
            for k in range(i-1, i+2):                 
                for l in range(j-1, j+2):                     
                    var += (pixel_array[k][l]-mean)**2                
            var /= 9                                      
            sd = math.sqrt(var)                                      
            image[i][j] = sd                  
            
    return image  

def computeThresholdGE(pixel_array, threshold_value,image_width, image_height):
    new_array = list()
    
    for i in range(image_height):
        row = list()
        for j in range(image_width):
            if(pixel_array[i][j] < threshold_value):
                row.append(0)
            else:
                row.append(255)
        new_array.append(row)
    return new_array

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    dilation = createInitializedGreyscalePixelArray(image_width, image_height)
                
    for r in range(image_height):
        for c in range(image_width):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if (r+i >= 0 and r+i < image_height and c+j >= 0 and c+j < image_width and pixel_array[r+i][c+j] != 0):
                        dilation[r][c] = 1
    return dilation

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(1, image_height - 1):
        for c in range(1, image_width - 1):
            if pixel_array[r][c] != 0 and pixel_array[r - 1][c] != 0 and pixel_array[r + 1][c] != 0:
                if pixel_array[r][c - 1] != 0 and pixel_array[r - 1][c - 1] != 0 and pixel_array[r + 1][c - 1] != 0:
                    if pixel_array[r][c + 1] != 0 and pixel_array[r - 1][c + 1] != 0 and pixel_array[r + 1][c + 1] != 0:
                        result[r][c] = 1
    return result

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
def bfs_traversal(pixel_array, visited, i, j, width, height, ccimg, count):
    q=Queue()

    x=[-1,0,1,0]
    y=[0,1,0,-1]
    
    number=0
    
    # add (i,j) into queue adn matk it as visited
    q.enqueue((i,j))
    visited[i][j]=True
    
    # do the following till queue becomes empty
    while(not q.isEmpty()):
        # take a position (a,b) from queue
        a,b=q.dequeue()
        # mark the nvalue at (a,b) in ccimg with component count
        ccimg[a][b]=count
        number+=1
        
        # if any unvisited 1 or 255 values is present in 4 sides of current posution, add it into queue
        for z in range(4):
            newI=a+x[z]
            newJ=b+y[z]
            if newI>=0 and newI<height and newJ>=0 and newJ<width and not visited[newI][newJ] and pixel_array[newI][newJ]!=0:
                visited[newI][newJ]=True
                q.enqueue((newI,newJ))
                
    # at last retun number of values in the current component
    return number


def computeConnectedComponentLabeling(pixel_array, width, height):
    # create visited array and ccimg array where size equal to width and height
    visited=[]
    ccimg=[]
    
    # make all the visited values as False and ccimg values as 0
    for i in range(height):
        temp1=[]
        temp2=[]
        for j in range(width):
            temp1.append(False)
            temp2.append(0)
        visited.append(temp1)
        ccimg.append(temp2)
    
    ccsizedict={}
    count=1
    
    # traverse pixel_array from left to right and from top to bottom
    for i in range(height):
        for j in range(width):
            # if any unvisited and 1 or 255 value pixel is found, then start bsf traversal from that value
            if not visited[i][j] and pixel_array[i][j]!=0:
                # get number of values in bfs traversal and add it into ccsizedict
                number=bfs_traversal(pixel_array, visited, i, j, width, height, ccimg, count)
                ccsizedict[count]=number
                count+=1
                
    
    return (ccimg, ccsizedict)

def computeEdgeMagnitude(horizontal, vertical, image_width, image_height):
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            mag = math.sqrt(math.pow(horizontal[i][j],2) + math.pow(vertical[i][j],2))
            
            new_array[i][j] = mag

    return new_array

def computeBoundaryBox(ccimg, ccsizes, image_height, image_width):
    # retrieve the label with the largest identified component
    largest_component_key = max(ccsizes, key=ccsizes.get) 
    final_array = createInitializedGreyscalePixelArray(image_width, image_height) # final image array
    
    # used to convert into list and find min values later for starting origin, and max for end values?
    x_vals = set()
    y_vals = set()
    
    # get the largest component
    for i in range(image_height): # ROWS (therefore 'y' value)
        if largest_component_key in ccimg[i]:
            final_array[i] = ccimg[i] # retrieve the row which has the largest component's pixels
            for j in range(image_width): #iterate through columns of row --> COLUMNS therefore 'x' value
                if ccimg[i][j] == largest_component_key: # if label of largest component found
                    x_vals.add(j)
                    y_vals.add(i)
    
    start_x = min(x_vals)
    start_y = min(y_vals)
    
    width = max(x_vals) - start_x
    height = max(y_vals) - start_y
    
    return start_x, start_y, width, height

# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode7"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    # STEP 1: Convert to greyscale and normalise
    greyscale = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    scaled_greyscale = scaleTo0And255AndQuantize(greyscale, image_width, image_height)
    
    # STEP 2: 2a: Option 1: Image gradient method
    # Apply a 3x3 Sobel filter in the x- and y-directions independently
    horizontal_edges = computeHorizontalEdgesSobelAbsolute(scaled_greyscale, image_width, image_height)
    vertical_edges = computeVerticalEdgesSobelAbsolute(scaled_greyscale, image_width, image_height)
    # take the absolute value of the difference between the results
    edge_magnitudes = computeEdgeMagnitude(horizontal_edges, vertical_edges, image_width, image_height)
    print("gradient done")

    # STEP 3: Gaussian filter and blurring 
    # for i in range(4):    
    #     gaussian_filtered_array = computeGaussianAveraging3x3RepeatBorder(stretched_array, image_width, image_height)
    # gaussian_filtered_array = computeGaussianAveraging3x3RepeatBorder(edge_magnitudes, image_width, image_height)
    for i in range(10):
        blurred_image = computeBoxAveraging3x3(edge_magnitudes, image_width, image_height) 
    for i in range(1):    
        gaussian_filtered_array = computeGaussianAveraging3x3RepeatBorder(blurred_image, image_width, image_height)
        
    repeat_contrast_stretch = scaleTo0And255AndQuantize(gaussian_filtered_array, image_width, image_height)
    axs1[0, 1].set_title('blurred_image of image')
    axs1[0, 1].imshow(repeat_contrast_stretch, cmap='gray')
    print("blurring done")
     
    # STEP 4: Threshold the image:
    convert_to_binary = computeThresholdGE(repeat_contrast_stretch, 92, image_width, image_height)
    
    # STEP 5: Erosion and dilation:
    morphological_close = computeDilation8Nbh3x3FlatSE(convert_to_binary, image_width, image_height)
    for i in range(9): 
        if i < 2:
            morphological_close = computeDilation8Nbh3x3FlatSE(morphological_close, image_width, image_height)
        else:
            morphological_close = computeErosion8Nbh3x3FlatSE(morphological_close, image_width, image_height)
    axs1[1, 0].set_title('morphological_close of image')
    axs1[1, 0].imshow(morphological_close, cmap='gray')
    print("Erosion/dilation done")

    # STEP 6: Connected component analysis
    (ccimg,ccsizes) = computeConnectedComponentLabeling(morphological_close,image_width,image_height)
    x, y, width, height = computeBoundaryBox(ccimg, ccsizes, image_height, image_width)

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    px_array = greyscale

    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    # rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
    #                  edgecolor='g', facecolor='none')
    rect = Rectangle((x, y), width, height, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()