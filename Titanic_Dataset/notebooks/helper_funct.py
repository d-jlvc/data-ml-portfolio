# ---------------------------------------
# Functions to reduce the number of lines
# in 'titanic_data_visualization.ipynb':
# ---------------------------------------


def title_distribution(df):
    
    title_mr = 0
    title_mrs = 0
    title_miss = 0
    
    for title_name in df['Title']:
        if title_name == 'Mr':
            title_mr += 1
        elif title_name == 'Mrs':
            title_mrs += 1
        else:
            title_miss += 1
            
    return f"Mr. {title_mr} | Mrs. {title_mrs} | Miss: {title_miss}"


def class_distribution(df):
    
    frs_class = 0
    snd_class = 0
    trd_class = 0
    
    for cls_type in df['Class']:
        if cls_type == 1:
            frs_class += 1
        elif cls_type == 2:
            snd_class += 1
        else:
            trd_class += 1
            
    # return frs_class, snd_class, trd_class
    return f"First class: {frs_class} | Second class: {snd_class} | Third class: {trd_class}"


def sex_distribution(df):
    
    males = 0
    females = 0
    
    for sex_type in df['Sex']:
        if sex_type == 'male':
            males += 1
        else:
            females += 1

    return f"Male passengers: {males} | Female passengers: {females}"


def familyStat_distribution(df):
    
    singles = 0
    families = 0
    
    for fam_type in df['FamilyStatus']:
        if fam_type == 'single':
            singles += 1
        else:
            families += 1
            
    return f"Single passengers: {singles} | Families: {families}"


def survived_distribution(df):
    
    survived = 0
    killed = 0
    
    for if_surv in df['Survived']:
        if if_surv == 'yes':
            survived += 1
        else:
            killed += 1
            
    return f"Survived: {survived} | Killed: {killed}"


def oldest_youngest(df):
    
    oldest_prsn = df.loc[df['Age'].idxmax()]
    youngest_prsn = df.loc[df['Age'].idxmin()]
    
    return f"""
>>. Oldest person: {oldest_prsn['Name']}, [{oldest_prsn['Age']:.0f}] - Survived: {oldest_prsn['Survived']}\n\
>>. Youngest person: {youngest_prsn['Name']}, [{youngest_prsn['Age']:.0f}] - Survived: {youngest_prsn['Survived']}
    """
    