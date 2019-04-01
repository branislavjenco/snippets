/**
 * Takes a Promise returning function and returns
 * a new function doesn't get called again after
 * it's been called the first time until it's resolved.
 * Something like lodash's debounce but not timed, but
 * based on the resolution of the promise
 * @param {function} fn that returns a promise
 */
export const onlyOnce = fn => {
  let runningPromise;
  let pending = false;
  return (...args) => {
    if (pending) {
      return runningPromise;
    } else {
      runningPromise = fn(...args);
      pending = true;
      return runningPromise.then(() => {
        pending = false;
        return runningPromise;
      });
    }
  };
};

/**
 * Thunkified onlyOnce to be used in action creators
 * @param {function} action creator
 * @returns {function} thunk version of action creator
 * that doesn't get called again until it's resolved
 */
export const onlyOnceAction = action => {
  const fo = onlyOnce((dispatch, actionArgs) =>
    dispatch(action(...actionArgs))
  );
  const thunk = (...actionArgs) => dispatch => fo(dispatch, actionArgs);
  return thunk;
};
