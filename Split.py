# Import necessary libraries
import xml.etree.ElementTree as ET
import os

# The folder where the parts are going to be stored in separate files  
output_folder = './ten_xml_parts'

# If the output folder does not exist, create it 
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# The input file in XML format
xml_file = 'ABSA16_Restaurants_Train_SB1_v2.xml'


# Parse the XML file
tree = ET.parse(xml_file) 
print(f'Parsing "{xml_file}"...')

# Get the root: "Reviews"
root = tree.getroot() 

# Iterate over the tree to know all the elements
#for e in root.iter():
  #print(e.tag)   # "Review", "Opinions", "Opinion", "sentence", "text"

# Discover the sub-elements/children in the root
#for child in root:   
  #print(child.tag, child.attrib)   # e.g. Review {"rid": "1004293"}

# Find all the reviews under the sub-element 'Review'
find_reviews = root.findall('Review') 

# Split the reviews in 10 parts with each part containing 35 reviews
split_parts = [find_reviews[i:i+ 35] for i in range(0, len(find_reviews), 35)]    
print('Splitting reviews in 10 parts...')
        
# Start counting from 1
index = 1

# For each of the ten parts contained in the already split reviews
for part in split_parts:
    print("Next XML please!")
    
    # Open and start creating multiple files in XML form
    full_path = output_folder + '/part{}.xml'.format(index)
    
    with open(full_path,'ab') as f:
        f.write(bytes('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n', 'utf-8'))
        f.write(bytes('<Reviews>\n', 'utf-8'))
        
        # Continue counting and change number of part
        index += 1       
        # Number of reviews per part
        cc = 0
        
        # For each element contained in each part
        for elem in part:
            cc +=1
            
            # Create a string representation of an XML element, including all sub-elements 
            elem = ET.tostring(elem, method='xml')  
            #print(elem)
            
            f.write(elem)
        print(cc)
        f.write(bytes("</Reviews>", 'utf-8'))
   
print('Success! The parts are stored in separate files.')
                

            
    
                   
                
    
          
            
