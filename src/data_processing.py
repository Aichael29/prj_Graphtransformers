import pandas as pd
import dgl
import torch
import time

def load_data():
    """
    Load OULAD dataset from CSV files.
    """
    print("Loading data...")
    start_time = time.time()
    assessments = pd.read_csv('data/raw/assessments.csv')
    courses = pd.read_csv('data/raw/courses.csv')
    student_assessments = pd.read_csv('data/raw/studentAssessment.csv')
    student_info = pd.read_csv('data/raw/studentInfo.csv')
    student_registration = pd.read_csv('data/raw/studentRegistration.csv')
    student_vle = pd.read_csv('data/raw/studentVle.csv')
    vle = pd.read_csv('data/raw/vle.csv')
    print(f"Data loaded in {time.time() - start_time:.2f} seconds.")
    return assessments, courses, student_assessments, student_info, student_registration, student_vle, vle


def preprocess_data():
    """
    Preprocess data to create a heterogeneous graph with edge features using DGL.
    """
    assessments, courses, student_assessments, student_info, student_registration, student_vle, vle = load_data()

    print("Cleaning and converting data...")
    # Ensure numeric IDs for all necessary columns
    student_info['id_student'] = student_info['id_student'].astype(int)
    student_assessments['id_student'] = student_assessments['id_student'].astype(int)
    student_assessments['id_assessment'] = student_assessments['id_assessment'].astype(int)
    student_registration['id_student'] = student_registration['id_student'].astype(int)
    vle['id_site'] = vle['id_site'].astype(int)
    student_vle['id_site'] = student_vle['id_site'].astype(int)
    student_vle['id_student'] = student_vle['id_student'].astype(int)

    print("Building relationships...")
    # Create mappings for relationships
    module_mapping = {code: idx for idx, code in enumerate(courses['code_module'].unique())}
    all_student_ids = set(student_registration['id_student']).union(student_assessments['id_student']).union(student_vle['id_student'])
    all_student_ids = sorted(all_student_ids)  # Ensure sorted order

    # Create a mapping for student IDs to graph node indices
    student_id_mapping = {student_id: idx for idx, student_id in enumerate(all_student_ids)}

    # Remove duplicate entries from student_info
    student_info = student_info.drop_duplicates(subset='id_student')
    print(f"Total unique students in student_info: {student_info['id_student'].nunique()}")

    # Debugging: Validate alignment
    print(f"Total students in graph: {len(all_student_ids)}")
    print(f"Total students in student_info: {student_info['id_student'].nunique()}")
    mismatched_ids = set(student_info['id_student']) - set(all_student_ids)
    print(f"Mismatched IDs (in student_info but not in graph): {len(mismatched_ids)}")
    print(f"Example mismatched IDs: {list(mismatched_ids)[:10]}")

    # Relationships and edge features
    relations = {
        ('student', 'registered_in', 'module'): (
            [student_id_mapping[s] for s in student_registration['id_student']],
            [module_mapping[m] for m in student_registration['code_module']]
        ),
        ('student', 'submitted', 'assessment'): (
            [student_id_mapping[s] for s in student_assessments['id_student']],
            student_assessments['id_assessment'].values.tolist()
        ),
        ('student', 'interacted_with', 'material'): (
            [student_id_mapping[s] for s in student_vle['id_student']],
            student_vle['id_site'].values.tolist()
        ),
        ('module', 'includes', 'assessment'): (
            [module_mapping[m] for m in assessments['code_module']],
            assessments['id_assessment'].values.tolist()
        ),
        ('module', 'uses', 'material'): (
            [module_mapping[m] for m in vle['code_module']],
            vle['id_site'].values.tolist()
        ),
    }

    # Validate node counts dynamically
    num_nodes_dict = {
        'student': len(all_student_ids),
        'module': len(module_mapping),
        'assessment': max(max(student_assessments['id_assessment']), max(assessments['id_assessment']), 0) + 1,
        'material': max(max(student_vle['id_site']), max(vle['id_site']), 0) + 1,
    }

    print("Node counts:")
    for node_type, count in num_nodes_dict.items():
        print(f"- {node_type}: {count} nodes")

    print("Creating DGL graph...")
    try:
        graph = dgl.heterograph({
            edge_type: (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64))
            for edge_type, (src, dst) in relations.items()
        }, num_nodes_dict=num_nodes_dict)
    except Exception as e:
        print(f"Error during graph creation: {e}")
        raise

    print("Adding node and edge features...")
    # Align student_info with graph node IDs
    aligned_student_info = pd.DataFrame(index=all_student_ids)  # Create DataFrame with graph IDs as index
    aligned_student_info = aligned_student_info.join(
        student_info.set_index('id_student'), how='left'
    )  # Align student_info to graph nodes

    # Debugging: Confirm alignment
    print(f"Before filling missing values, aligned student_info rows: {aligned_student_info.shape[0]}")

    # Fill missing values for nodes without features
    aligned_student_info.fillna({'gender': 'unknown', 'highest_education': 'unknown', 'age_band': 'unknown', 'disability': 'N'}, inplace=True)

    # Create one-hot encoded features
    student_features = pd.get_dummies(aligned_student_info[['gender', 'highest_education', 'age_band', 'disability']])

    # Ensure the number of features matches the number of nodes
    assert student_features.shape[0] == num_nodes_dict['student'], (
        f"Feature count ({student_features.shape[0]}) does not match node count ({num_nodes_dict['student']})!"
    )

    # Assign features to student nodes
    graph.nodes['student'].data['features'] = torch.tensor(student_features.values, dtype=torch.float32)

    # Add dummy features for modules, assessments, and materials if necessary
    graph.nodes['module'].data['features'] = torch.zeros((num_nodes_dict['module'], 1), dtype=torch.float32)
    graph.nodes['assessment'].data['features'] = torch.zeros((num_nodes_dict['assessment'], 1), dtype=torch.float32)
    graph.nodes['material'].data['features'] = torch.zeros((num_nodes_dict['material'], 1), dtype=torch.float32)

    print("Graph created successfully!")
    return graph


if __name__ == "__main__":
    start_time = time.time()
    try:
        graph = preprocess_data()
        dgl.save_graphs('data/processed/oulad_graph_with_features.bin', [graph])
        print(f"Graph saved successfully in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        print(f"Error: {e}")
