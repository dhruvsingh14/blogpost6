###################################
# PROJECT: Labor Market Outcomes  #
# PROGRAM: DataCombining.R        #
# PROGRAMMER: Dhruv Singh         #
# CHANGE LOG: 05/29/2020          #
###################################

#############################
# SET JOLTS DATA DIRECTORY  #
#############################
getwd()
setwd("C:/Dhruv/Misc/Personal/writing/Blogging/2_posts/2_August/wk3_post6/1_Post 6_blueprint/data/JOLTS")

#######################################
# IMPORT & CLEAN: JOB OPENINGS DATA   #
#######################################

## openings data
JobOpenings <- read.table(
  "JobOpenings.txt",
  sep="\t", header=TRUE)

# renaming columns
names(JobOpenings)[names(JobOpenings)=="value"] <- "value_jo"

# column add
JobOpenings$seasonal_adjustment <- substr(JobOpenings$series_id, 3, 3)
JobOpenings$industry <- substr(JobOpenings$series_id, 4, 9)
JobOpenings$region <- substr(JobOpenings$series_id, 10, 11)
JobOpenings$rate_level <- substr(JobOpenings$series_id, 14, 14)

# column drop: series id, footnotes
JobOpenings <- JobOpenings[-c(1,5)]

################################
# IMPORT & CLEAN: HIRES DATA   #
################################

## hires data
Hires <- read.table(
  "Hires.txt",
  sep="\t", header=TRUE)

# renaming columns
names(Hires)[names(Hires)=="value"] <- "value_hi"

# column add
Hires$seasonal_adjustment <- substr(Hires$series_id, 3, 3)
Hires$industry <- substr(Hires$series_id, 4, 9)
Hires$region <- substr(Hires$series_id, 10, 11)
Hires$rate_level <- substr(Hires$series_id, 14, 14)

# column drop: series id, footnotes
Hires <- Hires[-c(1,5)]

############################################
# IMPORT & CLEAN: TOTAL SEPARATIONS DATA   #
############################################

## separations data
TotalSeparations <- read.table(
  "TotalSeparations.txt",
  sep="\t", header=TRUE)

# renaming columns
names(TotalSeparations)[names(TotalSeparations)=="value"] <- "value_ts"

# column add
TotalSeparations$seasonal_adjustment <- substr(TotalSeparations$series_id, 3, 3)
TotalSeparations$industry <- substr(TotalSeparations$series_id, 4, 9)
TotalSeparations$region <- substr(TotalSeparations$series_id, 10, 11)
TotalSeparations$rate_level <- substr(TotalSeparations$series_id, 14, 14)

# column drop: series id, footnotes
TotalSeparations <- TotalSeparations[-c(1,5)]

################################
# IMPORT & CLEAN: QUITS DATA   #
################################

## quits data
Quits <- read.table(
  "Quits.txt",
  sep="\t", header=TRUE)

# renaming columns
names(Quits)[names(Quits)=="value"] <- "value_qu"

# column add
Quits$seasonal_adjustment <- substr(Quits$series_id, 3, 3)
Quits$industry <- substr(Quits$series_id, 4, 9)
Quits$region <- substr(Quits$series_id, 10, 11)
Quits$rate_level <- substr(Quits$series_id, 14, 14)

# column drop: series id, footnotes
Quits <- Quits[-c(1,5)]

##################################
# IMPORT & CLEAN: LAYOFFS DATA   #
##################################

## layoffs data
LayoffsDischarges <- read.table(
  "LayoffsDischarges.txt",
  sep="\t", header=TRUE)

# renaming columns
names(LayoffsDischarges)[names(LayoffsDischarges)=="value"] <- "value_dl"

# column add
LayoffsDischarges$seasonal_adjustment <- substr(LayoffsDischarges$series_id, 3, 3)
LayoffsDischarges$industry <- substr(LayoffsDischarges$series_id, 4, 9)
LayoffsDischarges$region <- substr(LayoffsDischarges$series_id, 10, 11)
LayoffsDischarges$rate_level <- substr(LayoffsDischarges$series_id, 14, 14)

# column drop: series id, footnotes
LayoffsDischarges <- LayoffsDischarges[-c(1,5)]


#################################
# IMPORT INDUSTRY LABELS DATA   #
#################################

## industry labels data
industry_labels <- read.table(
  "industry_labels.txt",
  sep="\t", header=TRUE)

# renaming columns
names(industry_labels)[names(industry_labels)=="industry_code"] <- "industry"
industry_labels$industry <- as.character(industry_labels$industry)
industry_labels <- industry_labels[-c(3:5)]

####################################
# MERGE AND REMOVE EXISTING DATA   #
####################################

m1 <- merge(JobOpenings, Hires, by = c("year", "period", "seasonal_adjustment", "industry", "region", "rate_level"))
rm(JobOpenings, Hires)

m2 <- merge(m1, TotalSeparations, by = c("year", "period", "seasonal_adjustment", "industry", "region", "rate_level"))
rm(m1, TotalSeparations)

m3 <- merge(m2, Quits, by = c("year", "period", "seasonal_adjustment", "industry", "region", "rate_level"))
rm(m2, Quits)

m4 <- merge(m3, LayoffsDischarges, by = c("year", "period", "seasonal_adjustment", "industry", "region", "rate_level"))
rm(m3, LayoffsDischarges)

combined_jolts_data <- merge(m4, industry_labels, by = "industry")
rm(m4, industry_labels)

#############################################
# SUBSET AND OUTPUT COMBINED JOLTS DATASET  #
#############################################

combined_jolts_data <- subset(combined_jolts_data, (seasonal_adjustment == "S" & rate_level == "L"))
combined_jolts_data$period <- gsub("M", "", combined_jolts_data$period)
combined_jolts_data <- combined_jolts_data[-c(4:6)]

write.csv(combined_jolts_data, "combined_jolts_data.csv")


###########################
# SET CES DATA DIRECTORY  #
###########################
setwd("C:/Dhruv/Misc/Personal/writing/Blogging/2_posts/2_August/wk3_post6/1_Post 6_blueprint/data/CES")

###############################
# IMPORT & CLEAN: CES  DATA   #
###############################

## ces data
CES <- read.table(
  "CES.txt",
  sep="\t", header=TRUE)

# subset
CES <- subset(CES, year > 1999)

# renaming columns
names(CES)[names(CES)=="value"] <- "value_earnings"

# column add
CES$seasonal_adjustment <- substr(CES$series_id, 3, 3)
CES$industry <- substr(CES$series_id, 4, 5)
CES$NAICS <- substr(CES$series_id, 6, 11)
CES$rate_level <- substr(CES$series_id, 12, 13)

# industry column for merge
CES$industry <- as.numeric(CES$industry)
CES$industry <- as.character(CES$industry)

# subsetting to seasonal, levels
CES <- subset(CES, (seasonal_adjustment == "S" & rate_level == "01" & NAICS == "000000"))
CES$period <- gsub("M", "", CES$period)

## industry labels data
industry_labels <- read.table(
  "industry_labels.txt",
  sep="\t", header=TRUE)

# renaming columns
names(industry_labels)[names(industry_labels)=="supersector_code"] <- "industry"
names(industry_labels)[names(industry_labels)=="supersector_name"] <- "industry_text"
industry_labels$industry <- as.character(industry_labels$industry)

# merging label
combined_ces_data <- merge(CES, industry_labels, by = "industry")
rm(CES, industry_labels)

###########################################
# SUBSET AND OUTPUT COMBINED CES DATASET  #
###########################################

# dropping unnecessary columns
combined_ces_data <- combined_ces_data[-c(1:2, 6:9)]
write.csv(combined_ces_data, "combined_ces_data.csv")


###################################
# MERGING JOLTS and CES DATASETS  #
###################################

labor_data <- merge(combined_ces_data, combined_jolts_data, by = c("year", "period", "industry_text"))
rm(combined_ces_data, combined_jolts_data)

# dropping industry column
labor_data <- labor_data[-c(5)]

######################################
# OUTPUTTING COMBINED LABOR DATASET  #
######################################

# resetting directory and writing out data
setwd("C:/Dhruv/Misc/Personal/writing/Blogging/2_posts/2_August/wk3_post6/1_Post 6_blueprint/data")

# turnover variables
x1 <- median(labor_data$value_qu) # quitting
x2 <- median(labor_data$value_dl) # layoffs
x3 <- median(labor_data$value_ts) # total separations

labor_data$turnover <- ""

for (i in 1:nrow(labor_data)) {
  if (labor_data$value_qu[i] > x1 & labor_data$value_dl[i] > x2 & labor_data$value_ts[i] > x3) {
    labor_data$turnover[i] <- "high_turnover"
  } else if (labor_data$value_qu[i] < x1 & labor_data$value_dl[i] < x2 & labor_data$value_ts[i] < x3) {
    labor_data$turnover[i] <- "low_turnover"
  } else {
    labor_data$turnover[i] <- "medium_turnover"
  }
} 

table(labor_data$turnover)

# high_turnover   low_turnover medium_turnover 
# 1273            1280             502 


# growth variables
x4 <- median(labor_data$value_jo)
x5 <- median(labor_data$value_hi)
x6 <- median(labor_data$value_earnings)

labor_data$growth <- ""

for (i in 1:nrow(labor_data)) {
  if (labor_data$value_jo[i] > x4 & labor_data$value_hi[i] > x5 & labor_data$value_earnings[i] > x6) {
    labor_data$growth[i] <- "high_growth"
  } else if (labor_data$value_jo[i] < x4 & labor_data$value_hi[i] < x5 & labor_data$value_earnings[i] < x6) {
    labor_data$growth[i] <- "low_growth"
  } else {
    labor_data$growth[i] <- "medium_growth"
  }
} 

table(labor_data$growth)

# high_growth    low_growth medium_growth 
# 1196          1207           652 


table(labor_data$turnover, labor_data$growth)

# high_growth low_growth medium_growth
# high_turnover          1027          1           245
# low_turnover              2       1032           246
# medium_turnover         167        174           161

# this makes perfect sense when thinking of the reward risk ratio

rm(x1, x2, x3, x4, x5, x6)

write.csv(labor_data, "labor_data.csv")

# basic summary stats

length(unique(labor_data$industry_text))
unique(labor_data$industry_text)
# [1] Construction                         Education and health services        Financial activities                
# [4] Government                           Information                          Leisure and hospitality             
# [7] Manufacturing                        Mining and logging                   Other services                      
#[10] Professional and business services   Retail trade                         Total private                       
#[13] Trade, transportation, and utilities

table(labor_data$industry_text[labor_data$turnover=="high_turnover" & labor_data$growth == "high_growth"])
# Professional and business services    235
# Total private                         235 
# Trade, transportation, and utilities  230


table(labor_data$industry_text[labor_data$turnover=="low_turnover" & labor_data$growth == "low_growth"])
# Financial activities  211                  
# Information           235        
# Mining and logging    235   

