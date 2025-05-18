from django import forms
from .models import Address

class AddressForm(forms.ModelForm):
    class Meta:
        model = Address
        fields = ['street_address', 'city', 'state', 'zip_code', 'country']
        labels = {
            'street_address': 'Street Address',
            'zip_code': 'ZIP Code'
        }