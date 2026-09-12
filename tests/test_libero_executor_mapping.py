from src.agent.libero_executor import DATASET_TASK_TO_SPATIAL_INDEX


def test_dataset_spatial_mapping_covers_selected_tasks():
    assert DATASET_TASK_TO_SPATIAL_INDEX[31] == 4
    assert DATASET_TASK_TO_SPATIAL_INDEX[39] == 9
    assert set(DATASET_TASK_TO_SPATIAL_INDEX.values()) == set(range(10))
