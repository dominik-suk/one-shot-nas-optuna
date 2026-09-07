from enum import Enum

PAMAP2_ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic walking", 9: "watching TV", 10: "computer work",
    11: "car driving", 12: "ascending stairs", 13: "descending stairs",
    16: "vacuum cleaning", 17: "ironing", 18: "folding laundry",
    19: "house cleaning", 20: "playing soccer", 24: "rope jumping",
}

class Pamap2ActivityType(Enum):
    ALL = 'all'
    PROTOCOL = 'protocol'
    ADL = 'adl'

    @property
    def valid_ids(self) -> list[int]:
        if self == Pamap2ActivityType.PROTOCOL:
            return [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
        if self == Pamap2ActivityType.ADL:
            return [1, 2, 3, 4, 12, 13]
        return list(PAMAP2_ACTIVITIES.keys())

    @property
    def valid_id_index_pair(self) -> dict[int, int]:
        return {activity_id: index for index, activity_id in enumerate(self.valid_ids)}

    @property
    def labels(self) -> list[str]:
        return [PAMAP2_ACTIVITIES[_id] for _id in self.valid_ids]
