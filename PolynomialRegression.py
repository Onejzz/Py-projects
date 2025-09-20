# Ice Cream Temperature and Sales

# Three lines to make our compiler able to draw:
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy

# Ice Cream Temperature
x = [-4.66226, -4.31656, -4.21398, -3.94966, -3.57855, -3.45571, -3.10844, -3.0813, -2.67246, -2.65229, -2.6515, -2.28826, -2.11187, -1.81894, -1.66035, -1.32638, -1.17313, -0.77333, -0.67375, -0.14963]

# Ice Cream Sales
y = [41.84298632027783, 34.66119537360234, 39.38300087682567, 37.53984488250128, 32.2845311879761, 30.00113847641735, 22.635401277012628, 25.36502221208036, 19.226970048254086, 20.27967917842273, 13.275828499002512, 18.123991212726547, 11.218294472789265, 10.012867848328882, 12.615181154152336, 10.957731335561812, 6.689122639625872, 9.3929686611095, 5.210162615266291, 4.673642540546473]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(min(x), max(x), 100)

# Predicting value of Ice Cream at -2.00 Degrees Celsius
bing_chilling = mymodel(-2)

print("-2Deg. Celsius Ice Cream Sales: ", bing_chilling)
print("R Squared: ", r2_score(y, mymodel(x)))

# Create the plot
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.xlabel("Ice Cream Temperature")
plt.ylabel("Ice Cream Sales")
plt.title("Ice Cream Temperature and Sales")

# Display the plot
plt.show()
