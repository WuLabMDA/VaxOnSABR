#-----function----
build_timevarying_table <- function(
    df,
    id_col = "MRN",
    t0_col = "TX_EndDate",
    time_col = "Time",
    event_col = "Event",
    vax_date_col = "Event_Date"
) {
  
  d <- df
  
  # Ensure Date format
  d[[t0_col]] <- as.Date(d[[t0_col]])
  d[[vax_date_col]] <- as.Date(d[[vax_date_col]])
  
  # Replace time = 0 with small epsilon
  eps <- 1e-6
  d[[time_col]][d[[time_col]] == 0] <- eps
  
  # Compute t_vax (months)
  d$t_vax <- as.numeric(d[[vax_date_col]] - d[[t0_col]]) / 30
  
  rows <- list()
  
  for (i in seq_len(nrow(d))) {
    
    r <- d[i, ]
    
    pid   <- r[[id_col]]
    stop  <- as.numeric(r[[time_col]])
    event <- as.numeric(r[[event_col]])
    t_vax <- r$t_vax
    
    # Case 1: no vaccine OR vaccine after follow-up
    if (is.na(t_vax) || t_vax >= stop) {
      
      rows[[length(rows) + 1]] <- data.frame(
        id = pid,
        start = 0,
        stop = stop,
        event = event,
        vax_tv = 0,
        vax_time = "No",
        interval = 1
      )
      
      next
    }
    
    # Case 2: vaccine before or at t0 (Pre-SABR)
    if (t_vax <= 0) {
      
      rows[[length(rows) + 1]] <- data.frame(
        id = pid,
        start = 0,
        stop = stop,
        event = event,
        vax_tv = 1,
        vax_time = "Pre",
        interval = 1
      )
      
      next
    }
    
    # Case 3: vaccine during follow-up → split
    
    # Pre-vaccine interval
    rows[[length(rows) + 1]] <- data.frame(
      id = pid,
      start = 0,
      stop = t_vax,
      event = 0,
      vax_tv = 0,
      vax_time = "No",
      interval = 1
    )
    
    # Post-vaccine interval
    rows[[length(rows) + 1]] <- data.frame(
      id = pid,
      start = t_vax,
      stop = stop,
      event = event,
      vax_tv = 1,
      vax_time = "Post",
      interval = 2
    )
  }
  
  # Combine all rows
  long_df <- do.call(rbind, rows)
  
  # Sort
  long_df <- long_df[order(long_df$id, long_df$start, long_df$stop), ]
  
  # Remove invalid intervals
  long_df <- long_df[long_df$stop > long_df$start, ]
  
  rownames(long_df) <- NULL
  
  return(long_df)
}