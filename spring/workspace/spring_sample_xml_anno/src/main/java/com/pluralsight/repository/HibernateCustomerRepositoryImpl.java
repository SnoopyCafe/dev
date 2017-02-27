package com.pluralsight.repository;

import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Repository;

import com.pluralsight.model.Customer;

@Repository("customerRepository")
public class HibernateCustomerRepositoryImpl implements CustomerRepository {
	
	/* (non-Javadoc)
	 * @see com.pluralsight.repository.CustomerRepository#findAll()
	 */
	@Override
	public List<Customer> findAll() {
		List<Customer> customers = new ArrayList<>();

		Customer customer1 = new Customer();
		customer1.setFirstname("Snoopy");
		customer1.setLastname("Brown");
		
		customers.add(customer1);
		
		Customer customer2 = new Customer();
		customer2.setFirstname("Charlie");
		customer2.setLastname("Brown");
		
		customers.add(customer2);

		return customers;
	}
}
