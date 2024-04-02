from pathlib import Path
import pandas as pd
import csv


class Item:
    pay_rate = 0.8 # The pay rate after 20% discount
    all = []
    def __init__(self, name : str, price : float, quantity= 0 ):
        # Run Validations to the receveid arguments
        assert price >= 0, f"Price {price} is not greater than zero"
        assert quantity >= 0, f"quantity {quantity} is not greater than zero"
        
        ## Asssign self to object
        self.name = name
        self.price = price
        self.quantity = quantity
        
        ## Append all the itms
        Item.all.append(self)
    
    def calculate_total_price(self):
        return self.price * self.quantity
    
    def apply_discout(self):
        self.price = self.price * self.pay_rate
    @classmethod    
    def instantiante_from_csv(cls):
        my_path= Path(__file__).cwd()
        df= pd.read_csv(my_path/"data" / "oop_phone_project"/"items.csv")
        reader = df.to_dict(orient='records')
        items= list(reader)
        for item in items:
            Item(
                 name= item.get('name'),
                 price= item.get('price'),
                 quantity= item.get('quantity'))
        
    def __repr__(self) -> str:
        return f"Item('{self.name}','{self.price}','{self.quantity}')"
        
    

############ Tests
## Test01    
item1 = Item("Laptop",100,3)
item2 = Item("Phone",100,4)  

# print(item1.calculate_total_price()) 
# print(item2.calculate_total_price())  
# item1.pay_rate= 0.5      
# print(item1.pay_rate) 
# print(item2.pay_rate)        
# ## Test02  

# print(item1.__dict__) ## give us the details of this instance

## Test03  
# # item2.pay_rate = 0.6
# # item1.apply_discout()
# # item2.apply_discout()

# # print(item1.price)
# # print(item2.price)

# # print(item2.all)

Item.instantiante_from_csv()
print(Item.all)