import pandas
names = {
    'firstnames' : ["Parul", "Akshita"],
    'lastnames':["Sinha", "Pandey"]
}
myTable = pandas.DataFrame(names)
print (myTable)