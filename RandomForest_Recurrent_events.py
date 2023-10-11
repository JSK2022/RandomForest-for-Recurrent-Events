import numpy as np
import pandas as pd


path="/Users/jeongsookim/Downloads"
data = pd.read_csv(f"{path}/simuDat.csv")

ids = data['id'].values
time_start = data['start'].values
time_stop = data['stop'].values
event = data['event'].values
x = data[['group','x1','gender']].values

### Riskset Counter ###
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

### Criterion using Pseudo-score test statistics ###
class PseudoScoreCriterion:
    def __init__(self, n_outputs, n_samples, unique_times, x, ids, time_start, time_stop, event):
        """
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


### TreeBuilder using PseudoScoreCriterion ###
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

        # Check depth and minimum ids required for split
        if self.max_depth is not None and depth >= self.max_depth:
            return {
                'feature': None,
                'threshold': None,
                'left_child': None,
                'right_child': None,
                'node_value': node_value,
                'unique_times': node_unique_times,
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

### Decision Tree for recurrent events data ###
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
        Predict the node_value of the terminal node for given samples.
        """

        # Ensure X is a list of samples
        X = np.array(X)

        mean_function_predictions = {}

        for sample_id in ids:
            samples_for_id = [X[i] for i, uid in enumerate(ids) if uid == sample_id]
            terminal_node_for_id = self.traverse_tree_for_id(samples_for_id, self.tree_)

            mean_function_predictions[sample_id] = terminal_node_for_id["node_value"]

        return mean_function_predictions

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

def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    tree_idx,
    n_trees,
    verbose=0,
    n_ids_bootstrap=None,
    random_state=None
):
    """
    Private function used to fit a single tree in parallel for recurrent events.
    """
    # Extract necessary data from y
    ids = y['id']
    time_start = y['time_start']
    time_stop = y['time_stop']
    event = y['event']

    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        unique_ids = np.unique(ids)
        n_ids = len(unique_ids)

        # Generate bootstrap samples using IDs
        sampled_ids = _generate_sampled_ids(
            RandomState(random_state), unique_ids, n_ids_bootstrap
        )

        # Expand sampled IDs to all their associated events
        indices = np.where(np.isin(ids, sampled_ids))[0]
        
        tree.fit(X[indices], ids[indices], time_start[indices], time_stop[indices], event[indices])
    else:
        tree.fit(X, ids, time_start, time_stop, event)
    
    return tree

### C-index for recurrent event data ###
def recurrent_concordance_index_score(predictions, X, event, ids):
    """
    Computes the modified C-statistic for recurrent events.
    
    :param predictions: Ensemble predictions for each observation.
    :param X: Feature data.
    :param event: Array indicating event occurrence.
    :param ids: Array of IDs for each observation.
    :return: C-statistic and prediction error.
    """
    unique_ids = np.unique(ids)
    n_unique_ids = len(unique_ids)

    # Calculate total events for each ID
    total_events = {uid: np.sum(event[ids == uid]) for uid in unique_ids}

    id_to_avg_prediction = {}
    for uid in unique_ids:
        uid_indices = np.where(ids == uid)[0]
        uid_predictions = [predictions[i] for i in uid_indices if i in predictions]

        # Check if the uid_predictions list is empty
        if not uid_predictions:
            continue

        # Calculate the average for each element
        max_length = max(map(len, uid_predictions))
        avg_prediction = []
        for i in range(max_length):
            avg_prediction.append(np.mean([pred[i] for pred in uid_predictions if i < len(pred)]))

        id_to_avg_prediction[uid] = avg_prediction

    concordant_pairs = 0
    permissible_pairs = 0

    for i in range(n_unique_ids):
        for j in range(i+1, n_unique_ids):
            uid_i = unique_ids[i]
            uid_j = unique_ids[j]
            
            if uid_i not in id_to_avg_prediction or uid_j not in id_to_avg_prediction:
                continue

            if total_events[uid_i] > total_events[uid_j]:
                permissible_pairs += 1
                if id_to_avg_prediction[uid_i][-1] > id_to_avg_prediction[uid_j][-1]:
                    concordant_pairs += 1

    c_index = concordant_pairs / permissible_pairs if permissible_pairs > 0 else 0
    prediction_error = 1 - c_index
    return c_index, prediction_error


### RandomForests for recurrent events ###
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

class RecurrentRandomForest(BaseEstimator):
    """
    A Random Forest model designed for recurrent event data.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_ids_split=2,
                 min_ids_leaf=1, bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False, max_ids=None,
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
        X = self._validate_data(X)  # This will validate X and ensure it's an array.
        self.n_features_in_ = X.shape[1]  # Set the number of features attribute.

        # Convert y to the required format
        y_converted = {
            'id': y['id'],
            'time_start': y['time_start'],
            'time_stop': y['time_stop'],
            'event': y['event']
        }

        # Get the number of bootstrap samples
        n_samples_bootstrap = _get_n_ids_bootstrap(len(np.unique(y['id'])), self.max_ids)

        # Train each tree in parallel
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_build_trees)(
                tree=tree,
                bootstrap=self.bootstrap,
                X=X,
                y=y_converted,
                tree_idx=i,
                n_trees=self.n_estimators,
                verbose=self.verbose,
                n_ids_bootstrap=n_samples_bootstrap,
                random_state=tree.random_state  # Pass the random state for each tree here
            ) for i, tree in enumerate(self.estimators_)
        )

        # Calculate OOB score and attributes if needed
        if self.oob_score:
            self._set_oob_score_and_attributes(X, y_converted)

        return self
    
    def _set_oob_score_and_attributes(self, X, y):
        """
        Calculates the out-of-bag (OOB) scores using the ensemble's predictions for the training data samples 
        that were not seen during the training of a given tree.
        Also sets the 'oob_prediction_' and 'oob_score_' attributes of the class.
        """
        X = self._validate_data(X)
        ids = y['id']
        event = y['event']

        all_predictions = {}

        for estimator in self.estimators_:
            sampled_ids = _generate_sampled_ids(estimator.random_state, np.unique(ids), self.max_ids)
            unsampled_ids = _generate_unsampled_ids(np.unique(ids), sampled_ids)

            # Refitting the tree using only the unsampled data.
            estimator.fit(X[unsampled_ids, :], y['id'][unsampled_ids], y['time_start'][unsampled_ids], y['time_stop'][unsampled_ids], y['event'][unsampled_ids])

            # Making predictions using the predict_mean_function
            p_estimator_result_all = estimator.predict_mean_function(x, ids)
            p_estimator_result = {uid: p_estimator_result_all[uid] for uid in unsampled_ids if uid in p_estimator_result_all}
        
            for uid, pred in p_estimator_result.items():
                if uid not in all_predictions:
                    all_predictions[uid] = []
                all_predictions[uid].append(pred)

        # Averaging the predictions
        averaged_predictions = {}
        for uid, preds in all_predictions.items():
            averaged_predictions[uid] = self._pad_and_average_predictions(preds, len(self.estimators_))

        self.oob_prediction_ = averaged_predictions

        # Calculating the C-index and Prediction Error using an external function
        self.oob_score_, self.oob_prediction_error_ = recurrent_concordance_index_score(averaged_predictions, X, event, ids)


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

    def _pad_and_average_predictions(self, all_predictions_for_id, n_trees):
        """
        Pad the predictions to the length of the longest prediction and then average them.
        """
        max_length = max(map(len, all_predictions_for_id))

        # Pad each prediction to the maximum length
        padded_predictions = []
        for prediction in all_predictions_for_id:
            if len(prediction) < max_length:
                pad_length = max_length - len(prediction)
                padded_prediction = np.concatenate([prediction, [prediction[-1]] * pad_length])
            else:
                padded_prediction = prediction
            padded_predictions.append(padded_prediction)

        # Average the padded predictions
        average_prediction = np.mean(padded_predictions, axis=0)
        return average_prediction.tolist()

    def predict_mean_function(self, X, ids):
        X = self._validate_X_predict(X)
    
        # Get predictions from each tree
        all_predictions = [tree.predict_mean_function(X, ids) for tree in self.estimators_]

        # Average the predictions for each unique ID
        averaged_predictions = {}
        for uid in np.unique(ids):
            uid_predictions = [tree_preds[uid] for tree_preds in all_predictions if uid in tree_preds]
            averaged_predictions[uid] = self._pad_and_average_predictions(uid_predictions, len(self.estimators_))
    
        return averaged_predictions

    def predict_rate_function(self, X, ids):
        X = self._validate_X_predict(X)
    
        # Get predictions from each tree
        all_predictions = [tree.predict_rate_function(X, ids) for tree in self.estimators_]

        # Average the predictions for each unique ID
        averaged_predictions = {}
        for uid in np.unique(ids):
            uid_predictions = [tree_preds[uid] for tree_preds in all_predictions if uid in tree_preds]
            averaged_predictions[uid] = self._pad_and_average_predictions(uid_predictions, len(self.estimators_))
    
        return averaged_predictions