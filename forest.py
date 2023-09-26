from Bio import Phylo
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
import pylab
from lxml import etree

def init_forest():
    # Create the root element
    root = ET.Element('phyloxml', attrib={'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                                      'xsi:schemaLocation': 'http://www.phyloxml.org http://www.phyloxml.org/1.10/phyloxml.xsd',
                                      'xmlns': 'http://www.phyloxml.org'})
    
    # Create an ElementTree object
    tree = ET.ElementTree(root)
    
    # Write the XML tree to a file
    with open('forest.xml', "wb") as file:
        tree.write(file)


def enter_cell(cell_id, time, time_offsets, in_frame):
    # Parse the XML file
    tree = ET.parse('forest.xml')
    root = tree.getroot()

    time_offsets.append(time)
    in_frame.append(cell_id)

    # Create the new phylogeny structure
    new_tree = Element('phylogeny', rooted="true")
    tree_name = SubElement(new_tree, 'name')
    tree_name.text = "Ecoli Population"
    
    # Create outer clade
    outer_clade = SubElement(new_tree, 'clade')

    # Second nested clade
    inner_clade = SubElement(outer_clade, 'clade', branch_length="{:.20f}".format(time))
    inner_clade_name = SubElement(inner_clade, 'name')
    inner_clade_name.text = f"{cell_id}"

    # Append the new phylogeny element
    root.append(new_tree)

    # Save the updated XML to the same file
    with open('forest.xml', "wb") as file:
        tree.write(file)


def exit_cell(cell_id, current_time, in_frame):
    # Parse the XML file
    tree = ET.parse('forest.xml')
    root = tree.getroot()

    in_frame.remove(cell_id)


    # Iterate through clade elements within the current phylogeny
    for clade_element in root.findall('.//clade'):
        
        # Check if the name is cell_id
        name_element = clade_element.getchildren()[0] 
        if name_element.text == str(cell_id):
            enter_time = float(clade_element.attrib.get('branch_length'))
            branch_length = "{:.20f}".format(current_time - enter_time)
            clade_element.set('branch_length', branch_length)
            
            # Save the updated XML to the same file
            tree.write('forest.xml', encoding="utf-8", xml_declaration=True)
            return


def duplicate_cell(cell_id, current_time, max_id, in_frame):
    # Parse the XML file
    tree = ET.parse('forest.xml')
    root = tree.getroot()
    
    # Parent branch stops
    in_frame.remove(cell_id)


    # Iterate through clade elements within the current phylogeny
    for clade_element in root.findall('.//clade'):
        
        # Check if the name is cell_id
        name_element = clade_element.getchildren()[0] 
        if name_element.text == str(cell_id):

            # Enter length 
            enter_time = float(clade_element.attrib.get('branch_length'))
            branch_length = "{:.20f}".format(current_time - enter_time)
            clade_element.set('branch_length', branch_length)

            # Create outer clade
            left_child = SubElement(clade_element, 'clade', branch_length=f"{current_time}")
            right_child = SubElement(clade_element, 'clade', branch_length=f"{current_time}")

            left_child_name = SubElement(left_child, 'name')
            right_child_name = SubElement(right_child, 'name')
            
            max_id += 1
            left_child_name.text = f"{max_id}"
            in_frame.append(max_id)

            max_id += 1
            right_child_name.text = f"{max_id}"
            in_frame.append(max_id)

            # Save the updated XML to the same file
            tree.write('forest.xml', encoding="utf-8", xml_declaration=True)
            return





def stop_forest(in_frame, stopping_time):
    # Parse the XML file
    tree = ET.parse('forest.xml')
    root = tree.getroot()
    print(in_frame)
    # Set the branch length for all remaining cells
    for clade_element in root.findall('.//clade'):
        # Check if the name is cell_id
        name_element = clade_element.getchildren()[0]
        if not name_element.text is None:    
            cell_id = int(name_element.text)
            if cell_id in in_frame:
                # Enter length
                enter_time = float(clade_element.attrib.get('branch_length'))
                print(f'id:{cell_id}')
                print(f'stopping_time:{stopping_time}')
                print(f'enter_time:{enter_time}')
                branch_length = "{:.20f}".format(stopping_time - enter_time)
                print(f'branch_length:{branch_length}')
                print(f'-------------------------------------------------------------------------')
                clade_element.set('branch_length', branch_length)

    # Save the changes back to the XML file
    tree.write('forest.xml', encoding="utf-8", xml_declaration=True) 

def prune_forest(config):
    # Read the XML data from the file with UTF-8 encoding
    with open('forest.xml', 'rb') as file:
        xml_data = file.read()

    # Parse the XML data
    root = etree.fromstring(xml_data)

    # XPath expression to select clade elements
    xpath_expr = ".//clade"

    # Find and remove the selected clades
    clades = root.xpath(xpath_expr, namespaces={'phy': 'http://www.phyloxml.org'})
    for clade in clades:
        branch_length = clade.get('branch_length')  # Get the branch_length attribute
        if branch_length and float(branch_length) <= config['duration_cutoff']:
            parent = clade.getparent()
            parent.remove(clade)

    # Convert the modified XML back to string
    modified_xml = etree.tostring(root, encoding='utf-8').decode('utf-8')

    # Write the modified XML back to the file
    with open('forest.xml', 'wb') as file:
        file.write(modified_xml.encode('utf-8'))




def save_forest(time_offsets, config):
    # Load the existing XML file as a string
    with open('forest.xml', 'r') as file:
        xml_string = file.read()

    # Replace 'phy:' with an empty string
    modified_xml_string = xml_string.replace('phy:', '')

    # Replace ':phy' with an empty string
    modified_xml_string = modified_xml_string.replace(':phy', '')

    # Save the updated XML to the same file
    with open('forest.xml', 'w') as file:
        file.write(modified_xml_string)



    trees = Phylo.parse('forest.xml', 'phyloxml')
    Phylo.draw(trees, time_offsets=time_offsets, do_show=False, config=config)
    pylab.savefig('forest.svg',format='svg', bbox_inches='tight', dpi=300)
