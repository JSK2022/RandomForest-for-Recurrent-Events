import numpy as np
import pandas as pd


path="/Users/jeongsookim/Downloads"
data = pd.read_csv(f"{path}/simuDat.csv")

ids = data['id'].values
time_start = data['time_start'].values
time_stop = data['time_stop'].values
event = data['event'].values
x = data[['group','x1','gender']].values

class RisksetCounter:
    def __init__(self, ids, time_start, time_stop, event):
        self.ids = ids
        self.time_start = time_start
        self.time_stop = time_stop
        self.event = event

        self.all_unique_times = np.unique(time_stop)
        self.n_unique_times = len(self.all_unique_times)

        self.n_at_risk = np.zeros(self.n_unique_times, dtype=np.int64)
        self.n_events = np.zeros(self.n_unique_times, dtype=np.int64)
        self.set_data()

        self.state_stack = []

    def set_data(self):
        unique_ids = set(self.ids)
        for t_idx, t in enumerate(self.all_unique_times):
            self.n_at_risk[t_idx] = sum([self.Y_i(id_, t_idx) for id_ in unique_ids])
            self.n_events[t_idx] = sum([self.dN_bar_i(id_, t_idx) for id_ in unique_ids])

    def Y_i(self, id_, t_idx):
        if t_idx >= len(self.all_unique_times):
            return 0
        time_at_t_idx = self.all_unique_times[t_idx]
        indices = (self.ids == id_) & (time_at_t_idx <= self.time_stop)
        return np.any(indices)

    def dN_bar_i(self, id_, t_idx):
        if t_idx >= len(self.all_unique_times):
            return 0
        time_at_t_idx = self.all_unique_times[t_idx]
        indices = (self.ids == id_) & (time_at_t_idx == self.time_stop) & (self.event == 1)
        return np.any(indices)

    def save_state(self):
        self.state_stack.append((self.ids.copy(), self.time_start.copy(), self.time_stop.copy(), self.event.copy(), self.n_at_risk.copy(), self.n_events.copy()))

    def load_state(self):
        if self.state_stack:
            self.ids, self.time_start, self.time_stop, self.event, self.n_at_risk, self.n_events = self.state_stack.pop()

    def update(self, new_ids, new_time_start, new_time_stop, new_event): 
        # Save the current state
        self.save_state()    

        # Compute the intersection of data
        mask = np.isin(self.ids, new_ids)
    
        # Extract data of the intersection
        updated_ids = self.ids[mask]
        updated_time_start = self.time_start[mask]
        updated_time_stop = self.time_stop[mask]
        updated_event = self.event[mask]

        # Update object variables based on the intersection data
        self.ids = updated_ids
        self.time_start = updated_time_start
        self.time_stop = updated_time_stop
        self.event = updated_event

        # Recalculate unique times based on the updated data
        self.all_unique_times = np.unique(np.concatenate([self.time_start, self.time_stop]))
        self.n_unique_times = len(self.all_unique_times)
    
        # Resize the n_at_risk and n_events arrays based on the updated unique times
        self.n_at_risk = np.zeros(self.n_unique_times, dtype=np.int64)
        self.n_events = np.zeros(self.n_unique_times, dtype=np.int64)

        # Update the n_at_risk and n_events arrays
        unique_ids = set(self.ids)  # Extract unique IDs to avoid redundant calculations
        for t_idx, t in enumerate(self.all_unique_times):
            self.n_at_risk[t_idx] = sum([self.Y_i(id_, t_idx) for id_ in unique_ids])
            self.n_events[t_idx] = sum([self.dN_bar_i(id_, t_idx) for id_ in unique_ids])

    def reset(self):
        self.load_state()

    def copy(self):
        return RisksetCounter(self.ids.copy(), self.time_start.copy(), self.time_stop.copy(), self.event.copy())

    def __reduce__(self):
        return (self.__class__, (self.ids, self.time_start, self.time_stop, self.event))

def argbinsearch(arr, key_val):
    arr_len = len(arr)
    min_idx = 0
    max_idx = arr_len

    while min_idx < max_idx:
        mid_idx = min_idx + ((max_idx - min_idx) // 2)

        if mid_idx < 0 or mid_idx >= arr_len:
            return -1

        mid_val = arr[mid_idx]
        if mid_val <= key_val:  # Change the condition to <=
            min_idx = mid_idx + 1
        else:
            max_idx = mid_idx

    return min_idx


class PseudoScoreCriterion:
    def __init__(self, n_outputs, n_samples, unique_times, x, ids, time_start, time_stop, event):
        """ㅊ
        Constructor of the class
        Initialize instance variables using the provided input parameters
        Objects 'riskset_left', 'riskset_right', and 'riskset_total' are initialized using the 'RisksetCounter' class
        """
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.unique_times = unique_times
        self.x = x
        self.ids = ids
        self.time_start = time_start
        self.time_stop = time_stop
        self.event = event

        self.unique_ids = set(self.ids)  # Store unique ids for later use
        self.unique_times = unique_times

        self.riskset_left = RisksetCounter(ids, time_start, time_stop, event)
        self.riskset_right = RisksetCounter(ids, time_start, time_stop, event)
        self.riskset_total = RisksetCounter(ids, time_start, time_stop, event)

        self.samples_time_idx = np.searchsorted(unique_times, time_stop)

        self.split_pos = 0
        self.split_time_idx = 0

        self._riskset_counter = RisksetCounter(ids, time_start, time_stop, event)  # 새로 추가

    def init(self, y, sample_weight, n_samples, samples, start, end):
        """
        Initialization function
        Reset the risk set counters ('riskset_left','riskset_right','riskset_total') and updates 'riskset_total' with new data
        """
        self.samples = samples
        self.riskset_left.reset()
        self.riskset_right.reset()
        self.riskset_total.reset()

        time_starts, stop_times, events = y[:, 0], y[:, 1], y[:, 2]
        ids_for_update = [self.ids[idx] for idx in samples[start:end]]
        time_starts_for_update = [time_starts[idx] for idx in samples[start:end]]
        stop_times_for_update = [stop_times[idx] for idx in samples[start:end]]
        events_for_update = [events[idx] for idx in samples[start:end]]

        # Combine unique times from both datasets
        self.unique_times = np.unique(np.concatenate([self.unique_times, stop_times_for_update]))

        self.riskset_total.update(ids_for_update, time_starts_for_update, stop_times_for_update, events_for_update)

    def set_unique_times(self, unique_times):
        """Sets the unique times for the current node."""
        self.unique_times = unique_times

## Group Indicator만으로 나누기...

    # Functions returning the risk set value and event value for the given ID and time index from the respective risk set (left or right)
    def Y_left_value(self, id_, t):
        return self.riskset_left.Y_i(id_, t)
    
    def Y_right_value(self, id_, t):
        return self.riskset_right.Y_i(id_, t)

    def dN_bar_left_value(self, id_, t):
        return self.riskset_left.dN_bar_i(id_, t)

    def dN_bar_right_value(self, id_, t):
        return self.riskset_right.dN_bar_i(id_, t)

    def temporary_update_riskset(self, riskset_counter, ids, time_start, time_stop, event):
        # Combine and find unique stop times from both nodes
        combined_time_stops = np.concatenate([self.riskset_left.time_stop, self.riskset_right.time_stop])
        unique_time_stops = np.unique(combined_time_stops)

        riskset_counter.all_unique_times = unique_time_stops

        # Resize the n_at_risk and n_events arrays based on the updated unique times
        riskset_counter.n_at_risk = np.zeros(len(unique_time_stops), dtype=np.int64)
        riskset_counter.n_events = np.zeros(len(unique_time_stops), dtype=np.int64)

        # Update the n_at_risk and n_events arrays
        unique_ids = set(ids)  # Extract unique IDs to avoid redundant calculations
        for t_idx, t in enumerate(unique_time_stops):
            riskset_counter.n_at_risk[t_idx] = sum([riskset_counter.Y_i(id_, t_idx) for id_ in unique_ids])
            riskset_counter.n_events[t_idx] = sum([riskset_counter.dN_bar_i(id_, t_idx) for id_ in unique_ids])

    def calculate_numerator(self):
        # Temporary update riskset
        self.temporary_update_riskset(self.riskset_left, self.riskset_left.ids, self.riskset_left.time_start, self.riskset_left.time_stop, self.riskset_left.event)
        self.temporary_update_riskset(self.riskset_right, self.riskset_right.ids, self.riskset_right.time_start, self.riskset_right.time_stop, self.riskset_right.event)
    
        w = (self.riskset_left.n_at_risk * self.riskset_right.n_at_risk) / (self.riskset_left.n_at_risk + self.riskset_right.n_at_risk)
        term = (self.riskset_left.n_events / self.riskset_left.n_at_risk) - (self.riskset_right.n_events / self.riskset_right.n_at_risk)
    
        return np.sum(w * term)

    def calculate_variance_estimate(self):
        """
        Update the variance estimate to be compatible with the provided function.
        """
    
        def var_comp(riskset, id_, uniTimeVec, w_const, max_w_const):
            """
            Compute the variance component for each observation, 
            similar to the var_comp function in the mcfDiff.test R code.
            """
            y_i_tj = np.array([riskset.Y_i(id_, t_idx) for t_idx in range(len(uniTimeVec))])
            yVec = riskset.n_at_risk
            n_i_tj = np.array([riskset.dN_bar_i(id_, t_idx) for t_idx in range(len(uniTimeVec))])
            dLambda = riskset.n_events / (riskset.n_at_risk + 1e-7)  # Avoid division by zero

            res_ij = np.where(yVec > 0, y_i_tj / yVec * (n_i_tj - dLambda), 0)

            max_res_ij = np.max(np.abs(res_ij))
    
            if max_res_ij > 0:
                re_res_ij = res_ij / max_res_ij
                reFactor = np.exp(np.log(max_res_ij) + np.log(max_w_const))
            else:
                re_res_ij = 0
                reFactor = 1
    
            res_const = (w_const / max_w_const) * re_res_ij

            return (np.sum(res_const) * reFactor) ** 2

        # Temporary update riskset
        self.temporary_update_riskset(self.riskset_left, self.riskset_left.ids, self.riskset_left.time_start, self.riskset_left.time_stop, self.riskset_left.event)
        self.temporary_update_riskset(self.riskset_right, self.riskset_right.ids, self.riskset_right.time_start, self.riskset_right.time_stop, self.riskset_right.event)

        # Extract required variables
        uniTimeVec = self.riskset_total.all_unique_times
        w_const = (self.riskset_left.n_at_risk * self.riskset_right.n_at_risk) / (self.riskset_left.n_at_risk + self.riskset_right.n_at_risk)
        max_w_const = np.max(w_const)

        # Calculate variance components for each ID in the left and right nodes
        varList1 = [var_comp(self.riskset_left, id_, uniTimeVec, w_const, max_w_const) 
                    for id_ in np.unique(self.riskset_left.ids)]

        varList2 = [var_comp(self.riskset_right, id_, uniTimeVec, w_const, max_w_const) 
                    for id_ in np.unique(self.riskset_right.ids)]
    
        # Sum the variance components
        varU_1 = np.sum(varList1)
        varU_2 = np.sum(varList2)
    
        return varU_1 + varU_2

    
    def calculate_denominator(self):
        return self.calculate_variance_estimate()

    def proxy_impurity_improvement(self):
        if len(self.riskset_left.n_at_risk) == 0 or len(self.riskset_right.n_at_risk) == 0:
            return -np.inf

        numer = self.calculate_numerator() ** 2
        denom = self.calculate_denominator()

        return numer / (denom + 1e-7)
    
    def update_riskset(self, ids_subset):
        # Update the riskset based on the subset of IDs at the current node
        unique_ids_subset = np.unique(ids_subset)
        self.riskset_counter.update(unique_ids_subset, self.time_start, self.time_stop, self.event)

    def node_value(self):
        """
        Returns the Nelson-Aalen estimator of the mean function μ(t) for the entities in the current node.
        """
        return self.node_value_from_riskset(self.riskset_total)

    def node_value_from_riskset(self, riskset_counter):
        """
        Returns the Nelson-Aalen estimator of the mean function μ(t) for the entities based on provided riskset_counter.
        """
        mu_hat_values = []
    
        # Initialize the cumulative sum of the Nelson-Aalen estimator
        cumsum_Nelson_Aalen = 0
    
        for t_idx, t in enumerate(self.unique_times):
            # Use n_at_risk and n_events from the riskset_counter
            n_at_risk_t = riskset_counter.n_at_risk[t_idx] if t_idx < len(riskset_counter.n_at_risk) else 0
            n_events_t = riskset_counter.n_events[t_idx] if t_idx < len(riskset_counter.n_events) else 0
            
            cumsum_Nelson_Aalen += n_events_t / (n_at_risk_t + 1e-7)  # Avoiding division by zero
            mu_hat_values.append(cumsum_Nelson_Aalen)
        
        return mu_hat_values

    # RisksetCounter의 상태를 저장하고 복원하기 위한 메서드를 추가합니다.
    def save_riskset_state(self):
        self._riskset_counter.save_state()

    def reset_riskset_state(self):
        self._riskset_counter.reset()

    def reset(self):
        """
        Functions to reset all risk set counters
        """
        self.riskset_total.reset()
        self.riskset_left.reset()
        self.riskset_right.reset()

    def copy(self):
        """
        Creates and returns a copy of the current object.
        """
        new_criterion = PseudoScoreCriterion(self.n_outputs, self.n_samples, self.unique_times,
                                             self.x, self.ids, self.time_start, self.time_stop, 
                                             self.event)
        new_criterion.riskset_left = self.riskset_left.copy()
        new_criterion.riskset_right = self.riskset_right.copy()
        new_criterion.riskset_total = self.riskset_total.copy()
        new_criterion.samples_time_idx = self.samples_time_idx.copy()
        if hasattr(self, 'samples'):
            new_criterion.samples = self.samples.copy()

        return new_criterion

def update_with_group_indicator(self, feature_index, group_indicator):
    """
    Update the criterion based on a specified feature and group indicator. 
    This will split the data into left and right nodes based on the provided feature and group indicator.
    """
    # Reset the riskset counters for the left and right nodes
    self.riskset_left.reset()
    self.riskset_right.reset()

    # Determine the split by the feature and group indicator
    left_mask = self.x[:, feature_index] <= group_indicator  # Changed to <= for continuous features
    right_mask = ~left_mask

    # Create empty lists to store the ids, start times, stop times, and events for both left and right splits
    ids_left, start_left, stop_left, event_left = [], [], [], []
    ids_right, start_right, stop_right, event_right = [], [], [], []

    # For each unique ID, decide whether to assign it to the left or right node based on the mask
    for id_ in self.unique_ids:
        id_indices = np.where(self.ids == id_)[0]  # Get all indices for this ID
        if left_mask[id_indices[0]]:
            ids_left.extend([self.ids[i] for i in id_indices])
            start_left.extend([self.time_start[i] for i in id_indices])
            stop_left.extend([self.time_stop[i] for i in id_indices])
            event_left.extend([self.event[i] for i in id_indices])
        else:
            ids_right.extend([self.ids[i] for i in id_indices])
            start_right.extend([self.time_start[i] for i in id_indices])
            stop_right.extend([self.time_stop[i] for i in id_indices])
            event_right.extend([self.event[i] for i in id_indices])

    # Set the all_unique_times for the risk sets of left and right nodes to the current node's unique times
    self.riskset_left.all_unique_times = self.unique_times
    self.riskset_right.all_unique_times = self.unique_times

    # Also, adjust the lengths of n_at_risk and n_events in both riskset_left and riskset_right to match unique_times
    self.riskset_left.n_at_risk = np.zeros(len(self.unique_times), dtype=np.int64)
    self.riskset_left.n_events = np.zeros(len(self.unique_times), dtype=np.int64)
    self.riskset_right.n_at_risk = np.zeros(len(self.unique_times), dtype=np.int64)
    self.riskset_right.n_events = np.zeros(len(self.unique_times), dtype=np.int64)

    # Update the risk sets for the left and right nodes
    self.riskset_left.update(ids_left, start_left, stop_left, event_left)
    self.riskset_right.update(ids_right, start_right, stop_right, event_right)

# 이 함수를 PseudoScoreCriterion 클래스에 추가합니다.
setattr(PseudoScoreCriterion, 'update', update_with_group_indicator)

# 추가로, left node와 right node의 데이터를 반환하는 메소드를 추가합니다.
def get_left_node_data(self):
    return self.riskset_left.ids, self.riskset_left.n_at_risk, self.riskset_left.n_events

def get_right_node_data(self):
    return self.riskset_right.ids, self.riskset_right.n_at_risk, self.riskset_right.n_events

setattr(PseudoScoreCriterion, 'get_left_node_data', get_left_node_data)
setattr(PseudoScoreCriterion, 'get_right_node_data', get_right_node_data)

def calculate_node_value_updated(self, side="left"):
    """
    Calculate the node value based on the updated RisksetCounter using get_left_node_data and get_right_node_data.
    
    Parameters:
        - side (str): Either "left" or "right" to determine which riskset to use for calculation.
    """
    if side == "left":
        ids, n_at_risk, n_events = self.get_left_node_data()
    elif side == "right":
        ids, n_at_risk, n_events = self.get_right_node_data()
    else:
        raise ValueError("Invalid side value. Expected 'left' or 'right'.")
    
    mask = np.isin(self.ids, ids)
    
    time_start_filtered = self.time_start[mask]
    time_stop_filtered = self.time_stop[mask]
    event_filtered = self.event[mask]
    
    riskset_temp = RisksetCounter(ids, time_start_filtered, time_stop_filtered, event_filtered)
    riskset_temp.n_at_risk = n_at_risk
    riskset_temp.n_events = n_events

    return self.node_value_from_riskset(riskset_temp)

# PseudoScoreCriterion 클래스에 위에서 정의한 함수를 추가합니다.
setattr(PseudoScoreCriterion, 'calculate_node_value', calculate_node_value_updated)



PseudoScoreCriterion





import pandas as pd
import numpy as np
from sklearn.utils import check_random_state

class PseudoScoreTreeBuilder:
    """
    Class designed to build a decision tree based on the pseudo-score test statistics criterion,
    typically used in recurrent events data analysis.
    """

    TREE_UNDEFINED = -1  # Placeholder

    def __init__(self, max_depth=None, min_ids_split=2, min_ids_leaf=1,
                 max_features=None, max_thresholds=None, min_impurity_decrease=0,
                 random_state=None):
        self.max_depth = max_depth
        self.min_ids_split = min_ids_split
        self.min_ids_leaf = min_ids_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = check_random_state(random_state)

    def split_indices(self, X_column, threshold, criterion, start, end):
        """Efficiently splits the data based on the given threshold for a specific feature column."""
        left_indices = np.where(X_column <= threshold)[0]
        right_indices = np.where(X_column > threshold)[0]

        # Convert local indices to global indices
        left_indices = np.arange(start, end)[left_indices]
        right_indices = np.arange(start, end)[right_indices]

        return left_indices, right_indices

    def _split(self, X, criterion, start, end):
        best_split = {
            'feature_index': None,
            'threshold': None,
            'improvement': -np.inf
        }
    
        # 가능한 스플릿 후보들을 저장하기 위한 리스트
        potential_splits = []

        n_features = X.shape[1]

        for feature_index in range(n_features):
            unique_thresholds = np.unique(X[start:end, feature_index])
            if len(unique_thresholds) <= 1:
                continue

            if self.max_thresholds and len(unique_thresholds) > self.max_thresholds:
                unique_thresholds = self.random_state.choice(unique_thresholds, self.max_thresholds, replace=False)

            for threshold in unique_thresholds:
                criterion.update(feature_index, threshold)
                improvement = criterion.proxy_impurity_improvement()

                left_indices, right_indices = self.split_indices(X[start:end, feature_index], threshold, criterion, start, end)

                # Ensure that both child nodes will have at least min_ids_leaf samples
                if len(left_indices) < self.min_ids_leaf or len(right_indices) < self.min_ids_leaf:
                    continue

                # self.min_impurity_decrease보다 큰 모든 스플릿 후보들을 저장
                if improvement > self.min_impurity_decrease:
                    potential_splits.append({
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'improvement': improvement
                    })

                if improvement > best_split['improvement']:
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'improvement': improvement
                    }

        return best_split, potential_splits



    
    def _build(self, X, y, criterion, depth=0, start=0, end=None):
        if end is None:
            end = X.shape[0]

        ids = y[start:end, 0]
        unique_ids = np.unique(ids)

        riskset_counter = RisksetCounter(ids, y[start:end, 1], y[start:end, 2], y[start:end, 3])
        node_value = criterion.node_value_from_riskset(riskset_counter)
        node_unique_times = riskset_counter.all_unique_times.tolist()
        node_value = node_value[:len(node_unique_times)]

        # We need to access n_at_risk and n_events as attributes, not as methods
        node_n_at_risk = riskset_counter.n_at_risk
        node_n_events = riskset_counter.n_events

        # Check depth and minimum ids required for split
        if self.max_depth is not None and depth >= self.max_depth:
            return {
                'feature': None,
                'threshold': None,
                'left_child': None,
                'right_child': None,
                'node_value': node_value,
                'unique_times': node_unique_times,
                'n_at_risk': node_n_at_risk.tolist(),  # Store as list for consistency
                'n_events': node_n_events.tolist(),    # Store as list for consistency
                'ids': unique_ids.tolist()
            }

        if len(unique_ids) < self.min_ids_split:
            return {
                'feature': None,
                'threshold': None,
                'left_child': None,
                'right_child': None,
                'node_value': node_value,
                'unique_times': node_unique_times,
                'n_at_risk': node_n_at_risk.tolist(),  # Store as list for consistency
                'n_events': node_n_events.tolist(),    # Store as list for consistency
                'ids': unique_ids.tolist()
            }

        best_split, potential_splits = self._split(X, criterion, start, end)

        for split in [best_split] + potential_splits:
            if split['threshold'] is None:
                continue

            left_indices, right_indices = self.split_indices(X[start:end, split['feature_index']], split['threshold'], criterion, start, end)

            # Check if there are enough unique ids in both left and right children after the split
            if len(np.unique(ids[left_indices])) >= self.min_ids_leaf and len(np.unique(ids[right_indices])) >= self.min_ids_leaf:
                best_split = split
                break
        else:  # No valid split found
            return {
                'feature': None,
                'threshold': None,
                'left_child': None,
                'right_child': None,
                'node_value': node_value,
                'unique_times': node_unique_times,
                'n_at_risk': node_n_at_risk.tolist(),  # Store as list for consistency
                'n_events': node_n_events.tolist(),    # Store as list for consistency
                'ids': unique_ids.tolist()
            }

        left_child = self._build(X[left_indices], y[left_indices], criterion, depth=depth+1)
        right_child = self._build(X[right_indices], y[right_indices], criterion, depth=depth+1)

        return {
            'feature': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left_child': left_child,
            'right_child': right_child,
            'node_value': node_value,
            'unique_times': node_unique_times,
            'n_at_risk': node_n_at_risk.tolist(),  # Store as list for consistency
            'n_events': node_n_events.tolist(),    # Store as list for consistency
            'ids': unique_ids.tolist()
        }

    def build(self, X, ids, time_start, time_stop, event):
        """
        The main method to invoke the tree building process.
        Initializes the pseudo-score criterion using the input data and constructs the tree using the _build method.
        """
        n_samples, n_features = X.shape
        y = np.c_[ids, time_start, time_stop, event]

        unique_times = np.unique(np.concatenate([time_start, time_stop]))
        criterion = PseudoScoreCriterion(n_outputs=1, n_samples=n_samples,
                                         unique_times=unique_times, x=X, ids=ids,
                                         time_start=time_start, time_stop=time_stop, event=event)

        tree = self._build(X, y, criterion)
        return tree
    

from sklearn.base import BaseEstimator

class RecurrentTree(BaseEstimator):
    def __init__(self, max_depth=None, min_ids_split=2, min_ids_leaf=1, 
                 max_features=None, max_thresholds=None, min_impurity_decrease=0,
                 random_state=None):
        """
        Constructor of the class
        Initializes the tree's hyperparameters and settings
        """
        self.max_depth = max_depth
        self.min_ids_split = min_ids_split
        self.min_ids_leaf = min_ids_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.tree_ = None

    def fit(self, X, ids, time_start, time_stop, event):
        """
        Trains the recurrent tree using the input data
        """
        X = np.array(X)
        ids = np.array(ids)
        time_start = np.array(time_start)
        time_stop = np.array(time_stop)
        event = np.array(event)

        # Use the PseudoScoreTreeBuilder to build the tree
        builder = PseudoScoreTreeBuilder(
            max_depth=self.max_depth,
            min_ids_split=self.min_ids_split,
            min_ids_leaf=self.min_ids_leaf,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state
        )
        self.tree_ = builder.build(X, ids, time_start, time_stop, event)
        return self

    def get_tree(self):
        """Return the tree as a dictionary."""
        return self.tree_

    def traverse_tree_for_id(self, X_id_samples, node):
        """
        Traverse the tree for a specific ID based on its samples.

        Args:
        - X_id_samples (list of arrays): The samples corresponding to a specific ID.
        - node (dict): The current node being evaluated in the tree.

        Returns:
        - node (dict): The terminal node for the specific ID.
        """
        if node["feature"] is None:  # Terminal node
            return node

        # Traverse the tree for each sample and collect the terminal nodes
        terminal_nodes = []
        for sample in X_id_samples:
            if sample[node["feature"]] <= node["threshold"]:
                terminal_nodes.append(self.traverse_tree_for_id([sample], node["left_child"]))
            else:
                terminal_nodes.append(self.traverse_tree_for_id([sample], node["right_child"]))

        # Check if all samples lead to the same terminal node
        first_terminal = terminal_nodes[0]
        if all(node == first_terminal for node in terminal_nodes):
            return first_terminal

        # If samples lead to different terminal nodes, it's ambiguous. For simplicity, return the first terminal node.
        # In a real-world scenario, this might need more sophisticated handling.
        return first_terminal

    def predict_mean_function(self, X, ids):
        """
        Predict the node_value, unique_times, n_at_risk, and n_events of the terminal node for given samples.
        """

        # Ensure X is a list of samples
        X = np.array(X)

        mean_function_predictions = {}

        for sample_id in ids:
            samples_for_id = [X[i] for i, uid in enumerate(ids) if uid == sample_id]
            terminal_node_for_id = self.traverse_tree_for_id(samples_for_id, self.tree_)

            # Get node_value, unique_times, n_at_risk, and n_events from the terminal node
            node_value = terminal_node_for_id.get("node_value",[])
            unique_times = terminal_node_for_id.get("unique_times", [])
            n_at_risk = terminal_node_for_id.get("n_at_risk", [])
            n_events = terminal_node_for_id.get("n_events", [])

            # Convert lists to dictionaries for easier access
            n_at_risk = {unique_times[i]: n_at_risk[i] for i in range(len(unique_times))}
            n_events = {unique_times[i]: n_events[i] for i in range(len(unique_times))}

            mean_function_predictions[sample_id] = {
                "mean_function": node_value,
                "unique_times": unique_times,
                "n_at_risk": n_at_risk,
                "n_events": n_events
            }

        return mean_function_predictions, unique_times, n_at_risk, n_events


    def predict_rate_function(self, X, ids):
        """
        Predict the rate function as the difference between unique time points for the terminal node.
        """
        
        if np.isscalar(ids):
            ids = [ids]
            
        mean_function_predictions = self.predict_mean_function(X, ids)
    
        rate_function_predictions = {}
        for sample_id in ids:
            mean_function_values = mean_function_predictions[sample_id]

            # Calculate rate function as the difference between consecutive mean function values
            rate_function = np.diff(mean_function_values, prepend=mean_function_values[0])

            rate_function_predictions[sample_id] = rate_function

        return rate_function_predictions
    
    def _map_terminal_nodes(self, node, current_id=[0]):
        """
        Recursively traverse the tree and assign unique integers to each terminal node.
        """
        if node["feature"] is None:  # Terminal node
            if "id" not in node:
                node["id"] = current_id[0]
                current_id[0] += 1
            return

        self._map_terminal_nodes(node["left_child"], current_id)
        self._map_terminal_nodes(node["right_child"], current_id)

    def apply(self, X, ids=None):
        """Return the index of the leaf that each unique ID is predicted as."""
        X = np.array(X, dtype=np.float32)
        if ids is None:
            ids = np.array([i for i in range(X.shape[0])])
        else:
            ids = np.array(ids)

        terminal_nodes = {}
        self._map_terminal_nodes(self.tree_)  # Reset the mapping

        for sample_id in np.unique(ids):
            samples_for_id = [X[i] for i, uid in enumerate(ids) if uid == sample_id]
            terminal_node_for_id = self.traverse_tree_for_id(samples_for_id, self.tree_)
            terminal_nodes[sample_id] = terminal_node_for_id["id"]

        return terminal_nodes
    

x = data[['group','gender']].values

import numpy as np
from numbers import Integral, Real
from sklearn.utils import check_random_state
from numpy.random import RandomState

def check_random_state(seed):
    """
    Check if seed is a valid random state.
    """
    if seed is None or isinstance(seed, int):
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.Generator):
        return seed
    else:
        raise ValueError(f"Invalid seed: {seed}")

def _get_n_ids_bootstrap(n_ids, max_ids):
    """
    Modified for recurrent events. Get the number of IDs in a bootstrap sample.
    """
    if max_ids is None:
        return n_ids

    if isinstance(max_ids, Integral):
        if max_ids > n_ids:
            msg = "`max_samples` must be <= n_ids={} but got value {}"
            raise ValueError(msg.format(n_ids, max_ids))
        return max_ids

    if isinstance(max_ids, Real):
        return max(round(n_ids * max_ids), 1)

def _generate_sampled_ids(random_state, unique_ids, max_ids):
    """
    Generate bootstrap sample indices based on unique IDs.
    """
    # Calculate the number of IDs to be sampled using the _get_n_ids_bootstrap function
    n_ids_bootstrap = _get_n_ids_bootstrap(len(unique_ids), max_ids)

    # Create a random instance with the given random_state
    random_instance = check_random_state(random_state)

    # Randomly select n_ids_bootstrap IDs from the unique_ids with replacement
    sampled_ids_indices = random_instance.choice(len(unique_ids), n_ids_bootstrap, replace=True)
    
    # Get the actual IDs using the indices
    sampled_ids = unique_ids[sampled_ids_indices]
    
    return sampled_ids

def _generate_unsampled_ids(unique_ids, sampled_ids):
    """
    Determine unsampled unique IDs from the entire set of IDs.
    """
    # 중복 제거된 sampled_ids
    unique_sampled_ids = np.unique(sampled_ids)
    
    # Find unsampled unique IDs
    unsampled_unique_ids = np.setdiff1d(unique_ids, unique_sampled_ids)
    return unsampled_unique_ids


from warnings import catch_warnings, simplefilter
from sklearn.utils.class_weight import compute_sample_weight

def _parallel_build_trees(tree, bootstrap, X, y, tree_idx, n_trees,
                                    verbose=0, n_ids_bootstrap=None, random_state=None, max_ids=None):
    """
    Private function used to fit a single tree in parallel with corrected y values extraction.
    """
    if verbose > 1:
        print("Building estimator %d of %d for this parallel run "
              "(total %d)..." % (tree_idx + 1, n_trees, n_trees))

    ids = y['id']
    time_start = y['time_start']
    time_stop = y['time_stop']
    event = y['event']

    # If bootstrap is True, generate a bootstrap sample for training
    if bootstrap:
        unique_ids = np.unique(ids)
        if isinstance(random_state, np.random.RandomState):
            rnd = random_state
        else:
            rnd = np.random.RandomState(random_state)
        
        sampled_ids = _generate_sampled_ids(rnd, unique_ids, max_ids)
        bootstrap_indices = np.where(np.isin(ids, sampled_ids))[0]
        X_bootstrap = X[bootstrap_indices]
        ids_bootstrap = ids[bootstrap_indices]
        time_start_bootstrap = time_start[bootstrap_indices]
        time_stop_bootstrap = time_stop[bootstrap_indices]
        event_bootstrap = event[bootstrap_indices]
        
        tree.fit(X_bootstrap, ids_bootstrap, time_start_bootstrap, time_stop_bootstrap, event_bootstrap)
    else:
        tree.fit(X, ids, time_start, time_stop, event)

    return tree


def generate_RE_nRE(data_dict):
    """
    Generate RE and nRE matrices from the given data.
    
    Parameters:
    - data_dict: A dictionary containing 'id', 'time_start', 'time_stop', and 'event' keys.
    
    Returns:
    - RE: A matrix with columns: [id, time_stop, cumulative_event_count].
    - nRE: An array containing total number of recurrent events for each ID.
    """
    import pandas as pd
    
    data_df = pd.DataFrame({
        'id': data_dict['id'],
        'time_start': data_dict['time_start'],
        'time_stop': data_dict['time_stop'],
        'event': data_dict['event']
    })
    
    data_sorted = data_df.sort_values(by=['id', 'time_start', 'time_stop'])
    data_sorted['cum_event'] = data_sorted.groupby('id')['event'].cumsum()
    
    RE = data_sorted[['id', 'time_stop', 'cum_event']].values
    nRE = data_sorted.groupby('id')['event'].sum().values
    
    return RE, nRE

def generate_cis(data):
    # Extract unique IDs
    unique_ids = np.unique(data['id'])

    # For each unique ID, get the last observed time
    common_observation_times = [np.max(data['time_stop'][data['id'] == id_]) for id_ in unique_ids]

    # Combine IDs and their corresponding common observation times
    Cis = np.column_stack((unique_ids, common_observation_times))
    
    return Cis

def est_cstat(score, N, Cis, RE, nRE):
    den = 0
    num = 0

    # Printing the keys in the score (which is essentially self.oob_prediction_)
    print(f"IDs in score: {list(score.keys())}")

    # Create a mapping from ID to its index in nRE
    unique_ids = np.unique(RE[:, 0].astype(int))
    id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}

    for i in range(N - 1):
        ID1 = int(Cis[i, 0])
        
        # Filter events happening before or at Cis[i, 2]
        search_event_indices = np.where(RE[:, 1] <= Cis[i, 1])[0]  # Use RE's second column (time)
        REtemp = RE[search_event_indices, :]
        
        # Count occurrences of each ID in REtemp
        unique_ids, counts = np.unique(REtemp[:, 0].astype(int), return_counts=True)
        
        # Check if the indices are valid (i.e., within bounds of N)
        valid_indices = [idx for idx in unique_ids if idx < N]
        valid_counts = counts[np.isin(unique_ids, valid_indices)]
        
        # Update nREc using the valid indices and counts
        nREc = np.zeros(N, dtype=int)
        nREc[valid_indices] = valid_counts

        # Sort the IDs for the remaining pairs
        IDpair = np.sort(Cis[(i+1):N, 0])
        
        # Printing IDpair before filtering
        print(f"IDpair before filtering: {IDpair}")
        
        IDpair = [id_ for id_ in IDpair if id_ in score and id_ != -1]
        
        # Printing IDpair after filtering
        print(f"IDpair after filtering: {IDpair}")
        
        indexed_IDpair = [id_to_index[id_] for id_ in IDpair if id_ in id_to_index]
        nREc = nREc[indexed_IDpair]

        lt_obs = np.where(nRE[ID1] < nREc, 1, 0)
        gt_obs = np.where(nRE[ID1] > nREc, 1, 0)

        last_value_from_score = lambda id_: score.get(id_, [0])[-1]
        score_values = np.array([last_value_from_score(id_) for id_ in IDpair])
        score_current = np.full(score_values.shape, last_value_from_score(ID1))
        
        lt_pred = np.where(score_current < score_values, 1, 0)
        gt_pred = np.where(score_current > score_values, 1, 0)

        den += np.sum(lt_obs + gt_obs)
        num += np.sum(lt_obs * lt_pred + gt_obs * gt_pred)

    estcstat = num / den if den != 0 else 0
    return estcstat

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.utils import check_random_state, check_array
from sklearn.exceptions import DataConversionWarning
from scipy.sparse import issparse
MAX_INT = np.iinfo(np.int32).max



from sklearn.utils import check_random_state
import numpy as np

def _generate_bootstrap_indices(tree, bootstrap, X, y, random_state, max_ids):
    """
    Private function used to generate bootstrap sample indices in parallel.
    """
    if bootstrap:
        unique_ids = np.unique(y['id'])
        if isinstance(random_state, np.random.RandomState):
            rnd = random_state
        else:
            rnd = np.random.RandomState(random_state)
        
        sampled_ids = _generate_sampled_ids(rnd, unique_ids, max_ids)
        bootstrap_indices = np.where(np.isin(y['id'], sampled_ids))[0]
        return bootstrap_indices
    else:
        return np.arange(len(y['id']))  # return all indices

def _fit_tree_with_bootstrap_samples(tree, X, y_converted, indices):
    """
    Fit a single tree with given bootstrap samples.
    """
    X_bootstrap = X[indices]
    y_bootstrap = {
        'id': y_converted['id'][indices],
        'time_start': y_converted['time_start'][indices],
        'time_stop': y_converted['time_stop'][indices],
        'event': y_converted['event'][indices]
    }
    tree.fit(X_bootstrap, y_bootstrap['id'], y_bootstrap['time_start'], y_bootstrap['time_stop'], y_bootstrap['event'])
    return tree

def _get_unsampled_bootstrap_indices(tree, bootstrap, X, y, random_state, max_ids):
    """
    Private function used to generate unsampled bootstrap sample indices in parallel.
    """
    if bootstrap:
        unique_ids = np.unique(y['id'])
        if isinstance(random_state, np.random.RandomState):
            rnd = random_state
        else:
            rnd = np.random.RandomState(random_state)
        
        sampled_ids = _generate_sampled_ids(rnd, unique_ids, max_ids)
        unsampled_ids = _generate_unsampled_ids(unique_ids, sampled_ids)
        unsampled_indices = np.where(np.isin(y['id'], unsampled_ids))[0]
        return unsampled_indices
    else:
        return np.array([])  # return an empty array for non-bootstrap cases


class RecurrentRandomForest(BaseEstimator):

    """
    A Random Forest model designed for recurrent event data.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_ids_split=2,
                 min_ids_leaf=1, bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False, max_ids=1.0,
                 min_impurity_decrease=0.0, max_features=None, max_thresholds=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_ids_split = min_ids_split
        self.min_ids_leaf = min_ids_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_ids = max_ids
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        
        # Initialize the random state for the forest
        self.random_state = check_random_state(random_state)
        
        # Create the estimators using the updated random states
        self.estimators_ = [self._make_estimator() for _ in range(self.n_estimators)]

    def _make_estimator(self):
        """
        Constructs a new instance of the 'RecurrentTree' with the specified hyperparameters.
        Allows for creating each tree with a different 'random_state' for randomness.
        """
        # Generate a new random state for each tree based on the forest's random state
        if isinstance(self.random_state, np.random.Generator):
            tree_random_state = self.random_state.integers(np.iinfo(np.int32).max)
        else:
            tree_random_state = self.random_state.randint(np.iinfo(np.int32).max)

        return RecurrentTree(
            max_depth=self.max_depth,
            min_ids_split=self.min_ids_split,
            min_ids_leaf=self.min_ids_leaf,
            random_state=tree_random_state,  # Pass the generated random state for the tree
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds
        )


    def fit(self, X, y):
        """
        Build the recurrent random forest.
        """
        # Validate the input data
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]
    
        # Convert y to the required format
        y_converted = {
            'id': y['id'],
            'time_start': y['time_start'],
            'time_stop': y['time_stop'],
            'event': y['event']
        }
    
        # If max_ids is None, set it to 1.0
        if self.max_ids is None:
            self.max_ids = 1.0

        # Get bootstrap indices for each tree in parallel
        bootstrap_indices_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_generate_bootstrap_indices)(
                tree=tree,
                bootstrap=self.bootstrap,
                X=X,
                y=y_converted,
                random_state=tree.random_state,
                max_ids=self.max_ids
            ) for tree in self.estimators_
        )
        
        # Get unsampled bootstrap indices for each tree in parallel
        unsampled_bootstrap_indices_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_get_unsampled_bootstrap_indices)(
                tree=tree,
                bootstrap=self.bootstrap,
                X=X,
                y=y_converted,
                random_state=tree.random_state,
                max_ids=self.max_ids
            ) for tree in self.estimators_
        )
        
        self.unsampled_bootstrap_indices_list_ = unsampled_bootstrap_indices_list

        # Train each tree using its respective bootstrap indices in parallel
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree_with_bootstrap_samples)(
                tree=tree,
                X=X,
                y_converted=y_converted,
                indices=indices
            ) for tree, indices in zip(self.estimators_, bootstrap_indices_list)
        )

            # Calculate OOB score and attributes if needed
        if self.oob_score:
            self._set_oob_score_and_attributes(X, y_converted)

        return self

    def _set_oob_score_and_attributes(self, X, y):
        """
        Calculate OOB score and attributes for each unsampled ID.
        """
        # OOB 예측값만 추출
        all_unsampled_indices = set()  # 모든 추정기를 걸쳐서 unsampled된 인덱스 수집

        for estimator, unsampled_indices in zip(self.estimators_, self.unsampled_bootstrap_indices_list_):
            # unsampled_indices를 all_unsampled_indices에 추가
            all_unsampled_indices.update(unsampled_indices)

        # unsampled_indices에 포함된 아이디만 추출
        unsampled_ids = np.unique(y['id'][list(all_unsampled_indices)])
    
        # 모든 아이디에 대해 predict_mean_function을 사용하여 예측
        mean_function_predictions = self.predict_mean_function(X, y['id'])

        self.oob_prediction_ = {}
        for uid in unsampled_ids:
            self.oob_prediction_[uid] = mean_function_predictions[uid]

        y_unsampled = {key: y[key][np.isin(y['id'], unsampled_ids)] for key in y}

        # unsampled 데이터를 사용하여 RE와 nRE 행렬 생성
        RE, nRE = generate_RE_nRE(y_unsampled)

        # unsampled 데이터를 사용하여 Cis 생성
        Cis = generate_cis(y_unsampled)

        N = len(Cis)

        # est_cstat 함수를 사용하여 C-index 계산
        self.oob_score_ = est_cstat(self.oob_prediction_, N, Cis, RE, nRE)
  


        """Validate input data('X') to ensure it's in the correct format and meets the necessary conditions for processing."""
    def _validate_data(self, X, accept_sparse=False, ensure_min_samples=1):
        """Validate input data('X') to ensure it's in the correct format and meets the necessary conditions for processing."""
        return check_array(X, accept_sparse=accept_sparse, ensure_min_samples=ensure_min_samples)
    
    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict."""
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Number of features of the model must match the input. Model n_features is {} and input n_features is {}."
                             .format(self.n_features_in_, X.shape[1]))
        return X

    def predict(self, X, ids):
        """
        Predict the mean function for the given samples using all trees in the forest.
        """
        # Ensure that the model has been trained
        if not hasattr(self, "estimators_"):
            raise NotFittedError("The model is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # Prepare a dictionary to store the results
        mean_function_predictions = {uid: {"mean_function": [], "unique_times": [], "n_at_risk": {}, "n_events": {}} for uid in ids}
    
        # Collect predictions from all trees
        all_tree_results = [tree.predict_mean_function(X, ids) for tree in self.estimators_]

        for tree_result, _, _, _ in all_tree_results:
            for uid in tree_result:
                result = tree_result[uid]
                mean_function_predictions[uid]["mean_function"].append(result["mean_function"])
                mean_function_predictions[uid]["unique_times"].extend(result["unique_times"])
                for time_point in result["n_at_risk"]:
                    mean_function_predictions[uid]["n_at_risk"][time_point] = mean_function_predictions[uid]["n_at_risk"].get(time_point, 0) + result["n_at_risk"][time_point]
                for time_point in result["n_events"]:
                    mean_function_predictions[uid]["n_events"][time_point] = mean_function_predictions[uid]["n_events"].get(time_point, 0) + result["n_events"][time_point]

        # Average the predictions for mean_function only
        n_trees = len(self.estimators_)
        for uid in mean_function_predictions:
            mean_function_predictions[uid]["mean_function"] = [sum(x) / n_trees for x in zip(*mean_function_predictions[uid]["mean_function"])]
            mean_function_predictions[uid]["unique_times"] = list(set(mean_function_predictions[uid]["unique_times"]))
            mean_function_predictions[uid]["unique_times"].sort()

        return mean_function_predictions

    def predict_mean_function(self, X, ids):
        """
        Predict the mean function for the given samples using all trees in the forest.
        """
        # Ensure that the model has been trained
        if not hasattr(self, "estimators_"):
            raise NotFittedError("The model is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
    
        # Use the `predict` method to get the predictions
        results = self.predict(X, ids)
    
        # Extract the mean_function from the results
        mean_functions = {uid: result["mean_function"] for uid, result in results.items()}
    
        return mean_functions

    def predict_rate_function(self, X, ids):
        """
        Predict the rate function for the given samples using all trees in the forest.
        """
        # mean_function을 예측
        mean_functions = self.predict_mean_function(X, ids)

        # 각 mean_function을 사용하여 rate function 계산
        rate_functions = {}
        for uid, mean_function_values in mean_functions.items():
            # 시간 간격의 차이를 사용하여 rate function 계산
            rate_function = np.diff(mean_function_values)
            rate_functions[uid] = rate_function

        return rate_functions


RecurrentRandomForest

def recurrent_permutation_importance(model, X, y, n_repeats=30, random_state=None):
    """
    주어진 RecurrentRandomForest 모델에 대한 특성의 Permutation Importance를 계산합니다.
    이 함수는 아이디별로 데이터를 섞되, 특성 행렬의 값은 섞인 아이디의 첫 번째 값으로 채웁니다.

    Parameters:
    - model: RecurrentRandomForest 모델
    - X: 입력 특성
    - y: 타겟
    - n_repeats: 중요도를 계산하기 위해 각 특성을 섞을 횟수.
    - random_state: 재현성을 위한 시드

    Returns:
    - importances: 각 특성의 Permutation Importance를 포함하는 2D 배열.
    - importances_mean: 각 특성의 중요도 평균.
    - importances_std: 각 특성의 중요도 표준편차.
    """
    
    random_state = check_random_state(random_state)
    
    # 원래 데이터를 사용하여 예측 수행
    baseline_preds = model.predict_mean_function(X, y['id'])
    RE, nRE = generate_RE_nRE(y)
    Cis = generate_cis(y)
    N = len(Cis)
    baseline_score = est_cstat(baseline_preds, N, Cis, RE, nRE)
    
    n_features = X.shape[1]
    unique_ids = np.unique(y['id'])
    importances = np.zeros((n_repeats, n_features))
    
    for feature_idx in range(n_features):
        for repeat in range(n_repeats):
            X_permuted = X.copy()
            
            # Shuffle data by ID
            shuffled_ids = random_state.permutation(unique_ids)
            for original_id, shuffled_id in zip(unique_ids, shuffled_ids):
                idx_original = np.where(y['id'] == original_id)[0]
                idx_shuffled = np.where(y['id'] == shuffled_id)[0]
                X_permuted[idx_original, feature_idx] = X[idx_shuffled[0], feature_idx]
            
            # 섞인 데이터를 사용하여 예측 수행
            preds_permuted = model.predict_mean_function(X_permuted, y['id'])
            score_permuted = est_cstat(preds_permuted, N, Cis, RE, nRE)
            
            # 특성의 중요도는 모델의 성능이 임의로 섞였을 때 얼마나 감소하는지를 기반으로 합니다.
            importances[repeat, feature_idx] = baseline_score - score_permuted
                
    importances_mean = importances.mean(axis=0)
    importances_std = importances.std(axis=0)
    
    return importances, importances_mean, importances_std